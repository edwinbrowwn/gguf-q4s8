#!/usr/bin/env python3
"""Select automatic BF16 Q4_0 -> Q8_0 promotions for any GGUF model.

The rank is a reconstruction-error heuristic. If an imatrix is supplied, its
per-column in_sum2 values weight the error. The input may be a single GGUF or
the first shard of a split GGUF; GGUFReader follows the metadata in the first
shard and exposes the complete tensor map.
"""
from __future__ import annotations
import argparse, math, re
from collections import Counter
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

LINE_RE = re.compile(r"^\[\s*\d+\s*/\s*\d+\]\s+(\S+)\s+-\s+\[([^\]]+)\],\s+type\s*=\s*(\S+),\s+size\s*=\s*([0-9.]+)\s+MiB(?:\s+->\s+[0-9.]+\s+MiB\s+\(([^)]+)\))?")
BLOCK_BYTES = {"q4_0": (32, 18), "q8_0": (32, 34)}
FLOAT_TYPES = {"f16": 2, "bf16": 2, "f32": 4}

def type_name(value: int) -> str:
    return GGMLQuantizationType(int(value)).name.lower()

def model_parts(first: Path) -> list[Path]:
    match = re.match(r"^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$", first.name)
    if not match:
        return [first]
    prefix, _index, count = match.groups()
    parts = sorted(first.parent.glob(f"{prefix}-?????-of-{count}.gguf"))
    if len(parts) != int(count):
        parts = sorted(first.parent.glob(f"{prefix}-*-of-{count}.gguf"))
    if len(parts) != int(count):
        raise SystemExit(f"expected {count} model shards for {first}, found {len(parts)}")
    return parts


def load_base_types(log: Path) -> dict[str, str]:
    result = {}
    for line in log.read_text(errors="replace").splitlines():
        m = LINE_RE.match(line)
        if m:
            name, _shape, source, _size, final = m.groups()
            result[name] = (final or source).lower()
    if not result:
        raise SystemExit(f"no tensor records found in {log}")
    return result

def tensor_bytes(shape: list[int], typ: str) -> int:
    n = math.prod(shape)
    if typ in BLOCK_BYTES:
        block, size = BLOCK_BYTES[typ]
        return 0 if shape[0] % block else (n // block) * size
    return n * FLOAT_TYPES[typ] if typ in FLOAT_TYPES else 0

def as_bf16_rows(tensor, ncols: int) -> np.ndarray:
    raw = np.asarray(tensor.data)
    if raw.dtype == np.uint8:
        words = raw.reshape(-1).view("<u2")
    elif raw.dtype == np.uint16:
        words = raw.reshape(-1).astype("<u2", copy=False)
    else:
        raise ValueError(f"{tensor.name}: expected BF16 bytes, got {raw.dtype}")
    if words.size % ncols:
        raise ValueError(f"{tensor.name}: values do not fit {ncols}-wide rows")
    return words.reshape(-1, ncols)

def error_for_chunk(words: np.ndarray, weights: np.ndarray | None) -> tuple[float, float]:
    x = (words.astype(np.uint32) << 16).view(np.float32)
    rows, ncols = x.shape
    blocks = x.reshape(rows, ncols // 32, 32)
    max_index = np.argmax(np.abs(blocks), axis=2)
    max_value = np.take_along_axis(blocks, max_index[..., None], axis=2)[..., 0]
    d4 = -max_value / 8.0
    inv4 = np.divide(1.0, d4, out=np.zeros_like(d4), where=d4 != 0)
    q4 = np.trunc(blocks * inv4[..., None] + 8.5).clip(0, 15)
    dq4 = (q4 - 8.0) * d4[..., None]
    d8 = np.max(np.abs(blocks), axis=2) / 127.0
    inv8 = np.divide(1.0, d8, out=np.zeros_like(d8), where=d8 != 0)
    q8 = np.rint(blocks * inv8[..., None]).clip(-128, 127)
    dq8 = q8 * d8[..., None]
    w = 1.0 if weights is None else (weights.reshape(1, ncols // 32, 32) if weights.ndim == 1 else weights.reshape(rows, ncols // 32, 32))
    return float(np.sum((blocks - dq4) ** 2 * w, dtype=np.float64)), float(np.sum((blocks - dq8) ** 2 * w, dtype=np.float64))

def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--bf16", required=True, type=Path)
    ap.add_argument("--base-log", required=True, type=Path)
    ap.add_argument("--imatrix", type=Path)
    ap.add_argument("--q8-fraction", type=float, default=10)
    ap.add_argument("--max-tensor-mib", type=float, default=0)
    ap.add_argument("--protect-low-bandwidth", action="store_true")
    ap.add_argument("--chunk-rows", type=int, default=1024)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    args = ap.parse_args()
    if not 0 <= args.q8_fraction <= 100: ap.error("--q8-fraction must be between 0 and 100")
    base = load_base_types(args.base_log)
    tensors = {}
    for part in model_parts(args.bf16):
        tensors.update({t.name: t for t in GGUFReader(str(part)).tensors})
    imatrix_tensors = {}
    if args.imatrix:
        imatrix_tensors = {t.name: t for t in GGUFReader(str(args.imatrix)).tensors}
    base_bytes = 0
    candidates, skipped = [], Counter()
    for name, typ in base.items():
        t = tensors.get(name)
        if t is None: skipped["missing"] += 1; continue
        shape = [int(x) for x in t.shape]
        size = tensor_bytes(shape, typ)
        if not size: skipped["unsupported"] += 1; continue
        if typ in BLOCK_BYTES: base_bytes += size
        if typ != "q4_0": continue
        if args.max_tensor_mib and tensor_bytes(shape, "q4_0") / 2**20 > args.max_tensor_mib: skipped["size"] += 1; continue
        if shape[0] % 32: skipped["shape"] += 1; continue
        ncols = shape[0]
        words = as_bf16_rows(t, ncols)
        imp = imatrix_tensors.get(name + ".in_sum2")
        importance = None
        if imp is not None:
            vals = np.asarray(imp.data, dtype=np.float32).reshape(-1)
            if vals.size == ncols: importance = vals
        e4 = e8 = 0.0
        for start in range(0, words.shape[0], max(1, args.chunk_rows)):
            a, b = error_for_chunk(words[start:start + args.chunk_rows], importance)
            e4 += a; e8 += b
        q4 = tensor_bytes(shape, "q4_0"); q8 = tensor_bytes(shape, "q8_0")
        low = bool(re.search(r"(?:attn_[kv]\.weight$|ssm|shared|shexp)", name, re.I))
        candidates.append({"name": name, "q4_bytes": q4, "q8_bytes": q8, "extra": q8-q4,
                           "q4_error": e4, "q8_error": e8, "gain": max(0, e4-e8),
                           "score": max(0, e4-e8)/max(1, q8-q4), "low": low,
                           "weighted": importance is not None})
    candidates.sort(key=lambda x: (x["score"], x["gain"]), reverse=True)
    selected, names, selected_q8, extra = [], set(), 0, 0
    def add(x):
        nonlocal selected_q8, extra
        selected.append(x); names.add(x["name"]); selected_q8 += x["q8_bytes"]; extra += x["extra"]
    if args.protect_low_bandwidth:
        for x in candidates:
            if x["low"]: add(x)
    for x in candidates:
        if x["name"] in names: continue
        if selected_q8 + x["q8_bytes"] <= args.q8_fraction / 100 * max(1, base_bytes + extra):
            add(x)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(f"^{re.escape(x['name'])}$=Q8_0" for x in selected) + ("\n" if selected else ""))
    args.plan.write_text("name\tshape\tq4_error\tq8_error\tgain\textra_bytes\tscore\tweighted\n" + "\n".join(f"{x['name']}\t\t{x['q4_error']:.9g}\t{x['q8_error']:.9g}\t{x['gain']:.9g}\t{x['extra']}\t{x['score']:.9g}\t{x['weighted']}" for x in selected) + "\n")
    frac = selected_q8 / max(1, base_bytes + extra)
    args.summary.write_text(f"Q8 fraction target: {args.q8_fraction:.3f}%\nSelected promotions: {len(selected)}\nEstimated Q8 fraction: {100*frac:.3f}%\nCandidates: {len(candidates)}\nImportance-weighted candidates: {sum(x['weighted'] for x in candidates)}\nSkipped: {dict(skipped)}\n")
    print(f"selected promotions: {len(selected)}")
    print(f"estimated Q8 fraction: {100*frac:.3f}%")
    return 0

if __name__ == "__main__": raise SystemExit(main())