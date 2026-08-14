# Method

## Pipeline

1. Start from BF16 GGUF, never an already-quantized model.
2. Select the first shard when the source is split.
3. Select deterministic calibration records from pinned datasets or reuse existing calibration text.
4. Run one `llama-imatrix` process per source with explicit accelerator devices, `-ngl 999`, layer split, and `--process-output`.
5. Merge source imatrices.
6. Run a Q4_0 dry run and parse its tensor map.
7. Rank Q4_0 candidates by BF16 reconstruction-error reduction from Q8_0, weighted by imatrix `in_sum2` values when available.
8. Emit a model-specific tensor override file under the immutable run directory.
9. Quantize with Q4_0 default, overrides, and merged imatrix. Split sources are written with `--keep-split`.
10. Validate output shards and preserve logs/manifest.

## GPU calibration command

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 llama-imatrix \
  --device ROCm0,ROCm1,ROCm2,ROCm3 \
  -m "$MODEL" -f "$CORPUS" -o "$OUTPUT" \
  -c 512 -b 512 -ub 512 --chunks "$CHUNKS" \
  --no-ppl --process-output -ngl 999 \
  -sm layer -ts 1/1/1/1 -fa on -np 1
```

The generic driver discovers devices by default. Explicit device names are recommended for reproducibility.

## Selection policy

Q4_0 is the base. The ranker estimates block reconstruction error directly from BF16 values. An imatrix's per-column `in_sum2` values are used as importance weights when their dimensions match. Promotions are constrained by `--q8-fraction`; `--protect-low-bandwidth` additionally reserves candidates matching attention, state, shared/output, embedding, or normalization naming patterns. The resulting policy is always saved and must be inspected for new architectures.

This is an engineering heuristic, not a universal quality guarantee. Validate with BF16-referenced KLD/PPL and downstream tests for every model family.

## Split GGUF

Pass the first source shard to both `build-q4s8-calibration.py` and `build-q4s8.sh`. The automatic ranker reads all source shards before computing scores. The standard Q4S8 deliverable is one merged GGUF even when the BF16 source is split; `--keep-split` is an explicit opt-in for numbered output shards.

## Validation

At minimum:

```bash
llama-quantize --dry-run "$MODEL" Q4_0
llama-cli --list-devices
```

Then compare BF16 and Q4S8 with the same corpus, context, batch, device split, and runtime flags. Report model size, tensor counts, KLD/PPL, same-top agreement, and throughput separately. Calibration GPU time must not be confused with CPU quantization time.

## Limitations

- Reconstruction-error ranking is not equivalent to downstream accuracy.
- `in_sum2` matching is architecture/tool dependent.
- Dataset quality and revision affect the result; revisions are recorded in the run manifest.
- MTP-aware collection requires fork-specific support and is enabled only when detected or explicitly requested.
- The default dataset adapters cover the three pinned NVIDIA datasets; arbitrary HF datasets may require an adapter.