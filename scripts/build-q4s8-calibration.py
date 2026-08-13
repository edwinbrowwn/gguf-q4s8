#!/usr/bin/env python3
"""Deterministic, one-command HF calibration and Q4S8 quantization for BF16 GGUF models.

The command downloads pinned HF revisions in parallel, samples locally with
stable hashes, deduplicates globally, creates disjoint calibration/held-out
sets, collects per-source GPU imatrices, merges them, automatically selects Q8 promotions, and quantizes BF16.
Runs are immutable: an existing run directory is never overwritten and no
artifact is removed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import itertools
import json
import math
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Any, Iterable

import ijson
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

TOOL_ROOT = Path(__file__).resolve().parents[1]
LLAMA_DEFAULT = os.environ.get("LLAMA_CPP_ROOT", str(Path.home() / "llama.cpp"))
DATASETS_DEFAULT = [
    "nvidia/Open-SWE-Traces",
    "nvidia/Nemotron-SFT-Math-v4",
    "nvidia/ChatQA2-Long-SFT-data",
]
REVISIONS_DEFAULT = str(TOOL_ROOT / "configs" / "three-dataset-revisions.json")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", value)


def stable_u64(seed: int, namespace: str, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).digest()[:8], "big")


def digest_text(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def clip(text: str, limit: int) -> str:
    text = text.replace("\x00", " ").strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.72)
    return text[:head] + "\n...[record clipped]...\n" + text[-(limit - head - 28):]


def serialize_messages(messages: Any, limit: int) -> str:
    parts = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            parts.append(as_text(msg))
            continue
        role = msg.get("role", "unknown")
        if msg.get("content"):
            parts.append(f"[{role}]\n{as_text(msg['content'])}")
        if msg.get("reasoning_content"):
            parts.append(f"[{role}:reasoning]\n{as_text(msg['reasoning_content'])}")
        if msg.get("tool_calls"):
            parts.append(f"[{role}:tool_calls]\n{as_text(msg['tool_calls'])}")
    return clip("\n\n".join(parts), limit)


def serialize_row(row: dict[str, Any], limit: int) -> str:
    if isinstance(row.get("trajectory"), list):
        return clip(
            f"repository: {row.get('repo', '')}\nlanguage: {row.get('language', '')}\n"
            + serialize_messages(row["trajectory"], limit), limit)
    if isinstance(row.get("messages"), list):
        return serialize_messages(row["messages"], limit)
    if "question" in row and ("answer" in row or "sub-paragraphs" in row):
        return clip("\n\n".join([
            "Context:", as_text(row.get("sub-paragraphs", "")),
            "Question:", as_text(row.get("question", "")),
            "Answer:", as_text(row.get("answer", "")),
        ]), limit)
    return clip(as_text({k: v for k, v in row.items() if k not in {"id", "uuid"}}), limit)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def iter_json_array(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as f:
        yield from ijson.items(f, "item")


def iter_parquet(path: Path) -> Iterable[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256):
        yield from batch.to_pylist()


def iter_file(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".parquet":
        return iter_parquet(path)
    if path.suffix == ".jsonl":
        return iter_jsonl(path)
    return iter_json_array(path)


def uid_for(dataset_id: str, row: dict[str, Any], index: int, component: str) -> str:
    for key in ("trajectory_id", "uuid", "id", "paragraph_id"):
        if row.get(key):
            return f"{dataset_id}:{component}:{row[key]}"
    return f"{dataset_id}:{component}:row-{index}"


def files_for(api: HfApi, dataset_id: str, revision: str) -> tuple[str, list[str]]:
    info = api.dataset_info(dataset_id, revision=revision)
    resolved = info.sha
    names = api.list_repo_files(dataset_id, repo_type="dataset", revision=resolved)
    if dataset_id == "nvidia/Open-SWE-Traces":
        files = sorted(x for x in names if (x.startswith("data/qwen35_openhands_trajectories/") or x.startswith("data/openhands_trajectories/")) and x.endswith(".parquet"))
    elif dataset_id == "nvidia/Nemotron-SFT-Math-v4":
        files = ["data/train.jsonl"]
    elif dataset_id == "nvidia/ChatQA2-Long-SFT-data":
        wanted = {"long_sft/long_sft_QA_train.json",
                  "NarrativeQA_131072/NarrativeQA_131072_QA_train.json"}
        files = sorted(x for x in names if x in wanted)
    else:
        files = sorted(x for x in names if x.endswith((".parquet", ".jsonl", ".json")) and "train" in x.lower())
        if not files:
            raise RuntimeError(f"Cannot discover a train file for {dataset_id}; pass a supported HF dataset")
    if not files:
        raise RuntimeError(f"No supported files found for {dataset_id}")
    return resolved, files


def download_one(repo: str, revision: str, filename: str, cache_root: Path) -> Path:
    target = cache_root / safe_name(repo) / revision / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    path = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                           repo_type="dataset", local_dir=str(cache_root / safe_name(repo) / revision))
    return Path(path)


def download_datasets(dataset_ids: list[str], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, list[Path]]:
    api = HfApi()
    specs: list[tuple[str, str, str]] = []
    pinned = json.loads(Path(args.revisions_file).read_text()) if args.revisions_file else {}
    for dataset_id in dataset_ids:
        requested_revision = pinned.get(dataset_id, args.revision)
        revision, files = files_for(api, dataset_id, requested_revision)
        manifest.setdefault("revisions", {})[dataset_id] = revision
        for filename in files:
            specs.append((dataset_id, revision, filename))
    print(f"Downloading {len(specs)} HF files with {args.download_workers} workers", flush=True)
    paths: dict[str, list[Path]] = {x: [] for x in dataset_ids}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {pool.submit(download_one, repo, rev, fn, Path(args.cache_root)): (repo, fn)
                   for repo, rev, fn in specs}
        for future in concurrent.futures.as_completed(futures):
            repo, filename = futures[future]
            path = future.result()
            paths[repo].append(path)
            print(f"downloaded {repo}:{filename} -> {path}", flush=True)
    for repo in paths:
        paths[repo].sort()
    manifest["downloaded_files"] = {repo: [str(x) for x in values] for repo, values in paths.items()}
    (Path(manifest["run_root"]) / "revisions.json").write_text(json.dumps(manifest["revisions"], indent=2) + "\n")
    return paths


def source_records(dataset_id: str, paths: list[Path]) -> Iterable[tuple[str, dict[str, Any]]]:
    for path in paths:
        component = path.parent.name
        if dataset_id == "nvidia/ChatQA2-Long-SFT-data" and "_dev" in path.name:
            continue
        for index, row in enumerate(iter_file(path)):
            yield uid_for(dataset_id, row, index, component), row


def add_heap(heap: list[tuple[int, str, str, dict[str, Any]]], item: tuple[int, str, str, dict[str, Any]], limit: int) -> None:
    score, uid, digest, record = item
    entry = (-score, uid, digest, record)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def collect(dataset_id: str, paths: list[Path], args: argparse.Namespace,
            global_digests: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    # Keep separate priority pools for Math COT/TIR and ChatQA long/Narrative.
    pools: dict[str, list[tuple[int, str, str, dict[str, Any]]]] = {}
    scanned = 0
    for uid, row in source_records(dataset_id, paths):
        scanned += 1
        text = serialize_row(row, args.max_record_chars)
        if not text:
            continue
        digest = digest_text(text)
        if digest in global_digests:
            continue
        component = "all"
        if dataset_id == "nvidia/Nemotron-SFT-Math-v4":
            component = str(row.get("subset", "unknown")).lower()
        elif dataset_id == "nvidia/ChatQA2-Long-SFT-data":
            component = "narrative" if "NarrativeQA" in uid else "long_sft"
        owner = global_digests.get(digest)
        if owner is not None and owner <= dataset_id:
            continue
        global_digests[digest] = dataset_id
        score = stable_u64(args.seed, f"{dataset_id}:{component}", uid)
        split = "heldout" if score % 1000 < int(args.holdout_fraction * 1000) else "calibration"
        pools.setdefault(component, [])
        add_heap(pools[component], (score, uid, digest, {"uid": uid, "text": text, "digest": digest, "split": split}), args.candidate_records)
    candidates = []
    for component, heap in pools.items():
        for neg_score, uid, digest, record in heap:
            record["score"] = -neg_score
            record["component"] = component
            candidates.append(record)
    return candidates, [], scanned


def choose(records: list[dict[str, Any]], dataset_id: str, chunks: int, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_chars = chunks * args.context_size * 4
    components = sorted({x["component"] for x in records})
    selected: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        rows = [x for x in records if x["component"] == component]
        # Equal component quotas prevent Math COT or ChatQA long_sft from dominating.
        component_chunks = chunks // len(components) + (component_index < chunks % len(components))
        component_target = max(args.context_size * 4, component_chunks * args.context_size * 4)
        cal = sorted((x for x in rows if x["split"] == "calibration"), key=lambda x: x["score"])
        dev = sorted((x for x in rows if x["split"] == "heldout"), key=lambda x: x["score"])
        total = 0
        for row in cal:
            selected.append(row)
            total += len(row["text"])
            if total >= component_target:
                break
        if total < component_target:
            raise RuntimeError(f"{dataset_id}/{component}: only {total} chars for {component_target}; increase --candidate-records")
        heldout.extend(dev)
    if not selected:
        raise RuntimeError(f"No calibration records selected for {dataset_id}")
    return selected, heldout


def write_corpus(path: Path, records: list[dict[str, Any]], dataset_id: str) -> dict[str, Any]:
    metadata = []
    chars = 0
    with path.open("w", encoding="utf-8") as out:
        for i, record in enumerate(records):
            out.write(f"\n\n===== {dataset_id} sample {i} id={record['uid']} =====\n{record['text']}\n")
            chars += len(record["text"])
            metadata.append({"index": i, "uid": record["uid"], "component": record["component"],
                            "content_sha256": record["digest"], "chars": len(record["text"])})
    return {"path": str(path), "records": metadata, "chars": chars}


def run(cmd: list[str], log: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, check=True, env=env)
    else:
        with log.open("w") as stream:
            subprocess.run(cmd, check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("dataset_ids", nargs="*", help="HF IDs; omit to use local Wikitext fallback")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output-root", default=str(Path.cwd() / "q4s8-runs"))
    ap.add_argument("--cache-root", default=str(Path.home() / ".cache" / "gguf-q4s8"))
    ap.add_argument("--revision", default="main", help="HF revision to resolve and pin")
    ap.add_argument("--revisions-file", default=REVISIONS_DEFAULT, help="JSON map of dataset IDs to pinned commit SHAs")
    ap.add_argument("--download-workers", type=int, default=3)
    ap.add_argument("--model", required=True, help="BF16 GGUF model; for split models pass the first shard")
    ap.add_argument("--llama-root", default=LLAMA_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--context-size", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--holdout-fraction", type=float, default=0.10)
    ap.add_argument("--candidate-records", type=int, default=4000)
    ap.add_argument("--max-record-chars", type=int, default=0,
                    help="record clip length; 0 chooses an automatic diversity cap")
    ap.add_argument("--min-records-per-dataset", type=int, default=64)
    ap.add_argument("--device", default=os.environ.get("GGUF_Q4S8_DEVICES", "auto"),
                    help="comma-separated GPU devices, or auto to use all detected accelerator devices")
    ap.add_argument("--tensor-split", default=None, help="tensor split passed to llama-imatrix/quant validation")
    ap.add_argument("--gpu-layers", default="999")
    ap.add_argument("--mtp", choices=("auto", "on", "off"), default="auto",
                    help="enable MTP-aware imatrix collection only when needed")
    ap.add_argument("--q8-fraction", type=float, default=10.0,
                    help="maximum final Q8_0 byte fraction for automatic promotion")
    ap.add_argument("--max-tensor-mib", type=float, default=0.0,
                    help="automatic ranker candidate size limit; 0 means unlimited")
    ap.add_argument("--no-protect-low-bandwidth", action="store_true")
    ap.add_argument("--quantize-threads", type=int, default=24)
    ap.add_argument("--output", default=None, help="final GGUF path or split-model base path")
    ap.add_argument("--imatrix", default=None, help="reuse an existing merged imatrix and skip collection")
    ap.add_argument("--wiki-fallback", action="store_true", help="use local Wikitext instead of the default HF datasets")
    ap.add_argument("--calibration-dir", default=None, help="reuse existing per-dataset calibration text files instead of downloading HF data")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--no-imatrix", action="store_true")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    has_hf = bool(args.dataset_ids)
    if args.calibration_dir:
        has_hf = False
    elif not has_hf and not args.wiki_fallback:
        args.dataset_ids = list(DATASETS_DEFAULT)
        has_hf = True
    args.batch_size = args.batch_size or 512
    args.iterations = args.iterations or (1000 if (has_hf or args.calibration_dir) else 100)
    if args.max_record_chars <= 0:
        # Keep each dataset represented by at least this many records, while
        # still allowing naturally short records to remain unmodified.
        total_chars = args.iterations * args.context_size * 4
        args.max_record_chars = max(4096, math.ceil(total_chars / (len(args.dataset_ids) or 1) / args.min_records_per_dataset))
    if not 0 < args.holdout_fraction < 1:
        ap.error("--holdout-fraction must be between 0 and 1")
    if args.calibration_dir:
        dataset_ids = [p.stem.removesuffix("-calibration") for p in sorted(Path(args.calibration_dir).glob("*-calibration.txt"))]
        if not dataset_ids:
            raise SystemExit(f"no *-calibration.txt files found in {args.calibration_dir}")
    elif not has_hf:
        dataset_ids = ["wikitext-fallback"]
    else:
        dataset_ids = list(dict.fromkeys(args.dataset_ids))
    if args.iterations < len(dataset_ids):
        ap.error("--iterations must be at least the number of datasets")

    root = Path(args.output_root) / args.run_id
    if root.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {root}")
    for name in ("raw", "datasets", "imatrices", "logs", "model"):
        (root / name).mkdir(parents=True)
    quotas = {dataset_id: args.iterations // len(dataset_ids) + (i < args.iterations % len(dataset_ids))
              for i, dataset_id in enumerate(dataset_ids)}
    manifest: dict[str, Any] = {
        "status": "downloading", "run_id": args.run_id, "dataset_ids": dataset_ids,
        "quotas": quotas, "iterations": args.iterations, "batch_size": args.batch_size,
        "context_size": args.context_size, "seed": args.seed, "holdout_fraction": args.holdout_fraction,
        "revision_requested": args.revision, "revisions_file": args.revisions_file,
        "cache_root": args.cache_root, "model": args.model, "run_root": str(root),
        "effective_max_record_chars": args.max_record_chars,
        "min_records_per_dataset": args.min_records_per_dataset,
        "device_requested": args.device, "tensor_split": args.tensor_split,
        "gpu_layers": args.gpu_layers, "mtp_mode": args.mtp,
        "q8_fraction": args.q8_fraction, "imatrix_requested": args.imatrix,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.calibration_dir:
        paths = {dataset_id: [Path(args.calibration_dir) / f"{dataset_id}-calibration.txt"] for dataset_id in dataset_ids}
        manifest["downloaded_files"] = {k: [str(v[0])] for k, v in paths.items()}
    elif has_hf:
        paths = download_datasets(dataset_ids, args, manifest)
    else:
        paths = {}
        wiki = Path(args.llama_root) / "wikitext-2-raw" / "wiki.train.raw"
        if not wiki.is_file():
            raise SystemExit(f"Wikitext fallback is missing: {wiki}")
        paths["wikitext-fallback"] = [wiki]
        manifest["downloaded_files"] = {"wikitext-fallback": [str(wiki)]}
    manifest["status"] = "sampling"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    global_digests: dict[str, str] = {}
    specs: list[tuple[str, int, Path]] = []
    for dataset_id in dataset_ids:
        chunks = quotas[dataset_id]
        name = safe_name(dataset_id)
        calibration_path = root / "datasets" / f"{name}-calibration.txt"
        heldout_path = root / "datasets" / f"{name}-heldout.txt"
        if args.calibration_dir:
            # Existing corpora are already deterministically selected. Reuse
            # them byte-for-byte instead of treating the whole file as one
            # record and clipping/reselecting it.
            source = paths[dataset_id][0]
            shutil.copyfile(source, calibration_path)
            heldout_path.write_text("")
            cmeta = {"path": str(calibration_path), "source": str(source),
                     "reused": True, "chars": calibration_path.stat().st_size}
            hmeta = {"path": str(heldout_path), "reused": True, "chars": 0}
            manifest.setdefault("datasets", {})[dataset_id] = {
                "scanned_records": None, "selected_calibration": None,
                "selected_heldout": 0, "calibration": cmeta, "heldout": hmeta,
            }
            specs.append((name, chunks, calibration_path))
            print(f"{dataset_id}: reused={source} chunks={chunks}", flush=True)
            continue
        if dataset_id == "wikitext-fallback":
            text = Path(paths[dataset_id][0]).read_text(encoding="utf-8")
            records = [{"uid": "wiki.train.raw", "text": text, "digest": digest_text(text), "component": "all", "split": "calibration", "score": 0}]
            scanned = 1
        else:
            records, _, scanned = collect(dataset_id, paths[dataset_id], args, global_digests)
        calibration, heldout = choose(records, dataset_id, chunks, args)
        cmeta = write_corpus(calibration_path, calibration, dataset_id)
        hmeta = write_corpus(heldout_path, heldout, dataset_id)
        manifest.setdefault("datasets", {})[dataset_id] = {
            "scanned_records": scanned, "selected_calibration": len(calibration),
            "selected_heldout": len(heldout), "calibration": cmeta, "heldout": hmeta,
        }
        specs.append((name, chunks, calibration_path))
        print(f"{dataset_id}: scanned={scanned} selected={len(calibration)} heldout={len(heldout)} chunks={chunks}", flush=True)
    manifest["status"] = "prepared"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(f"Prepared run: {root}")
        return 0

    bin_dir = Path(args.llama_root) / "build/bin"
    env = os.environ.copy()
    def detected_devices() -> list[str]:
        probe = subprocess.run([str(bin_dir / "llama-imatrix"), "--list-devices"],
                               check=True, text=True, capture_output=True, env=env)
        devices = []
        for line in probe.stdout.splitlines():
            m = re.match(r"^\s+([A-Za-z]+[0-9]+):", line)
            if m and m.group(1).lower() not in {"cpu0", "cpu"}:
                devices.append(m.group(1))
        if not devices:
            raise RuntimeError("--device auto found no accelerator devices; pass --device explicitly")
        return devices
    devices = detected_devices() if args.device == "auto" else [x.strip() for x in args.device.split(",") if x.strip()]
    if not devices:
        raise SystemExit("--device must name at least one accelerator")
    tensor_split = args.tensor_split or ",".join("1" for _ in devices)
    mtp_enabled = args.mtp == "on"
    if args.mtp == "auto":
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(args.llama_root) / "gguf-py"))
            from gguf import GGUFReader
            part_match = re.match(r"^(.*)-[0-9]{5}-of-([0-9]{5})\.gguf$", Path(args.model).name)
            if part_match:
                prefix, count = part_match.groups()
                model_parts = sorted(Path(args.model).parent.glob(f"{prefix}-?????-of-{count}.gguf"))
            else:
                model_parts = [Path(args.model)]
            mtp_enabled = any("nextn" in t.name.lower() for part in model_parts for t in GGUFReader(str(part)).tensors)
        except Exception:
            mtp_enabled = False
    if mtp_enabled:
        env["LLAMA_IMATRIX_PROCESS_MTP"] = "1"
    matrices: list[Path] = []
    if args.imatrix:
        merged = Path(args.imatrix).resolve()
        if not merged.is_file():
            raise SystemExit(f"existing imatrix does not exist: {merged}")
        manifest["imatrix_reused"] = str(merged)
    elif not args.no_imatrix:
        for name, chunks, corpus in specs:
            matrix = root / "imatrices" / f"imatrix-{name}.gguf"
            run([str(bin_dir / "llama-imatrix"), "-m", args.model, "-f", str(corpus), "-o", str(matrix),
                 "-c", str(args.context_size), "-b", str(args.batch_size), "-ub", str(args.batch_size),
                 "--chunks", str(chunks), "--no-ppl", "--process-output", "-ngl", str(args.gpu_layers), "-sm", "layer",
                 "--device", ",".join(devices), "-ts", tensor_split, "-fa", "on", "-np", "1"], root / "logs" / f"imatrix-{name}.log", env)
            matrices.append(matrix)
        merged = root / "imatrices" / "imatrix-merged.gguf"
        merge_cmd = [str(bin_dir / "llama-imatrix"), "-m", args.model,
                     "--in-file", ",".join(str(matrix) for matrix in matrices),
                     "-o", str(merged)]
        run(merge_cmd, root / "logs" / "imatrix-merge.log")
    else:
        merged = root / "imatrices" / "imatrix-merged.gguf"

    if not args.no_quantize:
        builder = TOOL_ROOT / "scripts" / "build-q4s8.sh"
        output_model = Path(args.output) if args.output else root / "model" / f"{safe_name(Path(args.model).stem)}-{args.run_id}-Q4S8.gguf"
        build_cmd = [str(builder), "--repo", args.llama_root, "--build-dir", str(bin_dir.parent),
                     "--model", args.model, "--output", str(output_model),
                     "--q8-fraction", str(args.q8_fraction), "--threads", str(args.quantize_threads)]
        if not args.no_imatrix:
            build_cmd += ["--imatrix", str(merged)]
        if args.max_tensor_mib:
            build_cmd += ["--max-tensor-mib", str(args.max_tensor_mib)]
        if not args.no_protect_low_bandwidth:
            build_cmd += ["--protect-low-bandwidth"]
        run(build_cmd, root / "logs" / "quantize.log")
        manifest["quantized_model"] = str(output_model)
        manifest["tensor_type_overrides"] = str(output_model.parent / (output_model.stem + ".tensor-types.txt"))
    manifest["merged_imatrix"] = str(merged)
    manifest["status"] = "complete"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed run: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())