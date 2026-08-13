# gguf-q4s8

`gguf-q4s8` is a reproducible, model-agnostic mixed-precision GGUF workflow. It starts from a BF16 GGUF, uses calibration data to generate an importance matrix on GPU, keeps Q4_0 as the base, and automatically promotes selected tensors to Q8_0 under a byte budget. Q4S8 is a recipe, not a new GGUF tensor type.

## One-command workflow

Build or use a llama.cpp tree with `llama-imatrix` and `llama-quantize`, install the Python dependencies, then run:

```bash
python3 scripts/build-q4s8-calibration.py \
  --model /models/Muse-Glimmer-30B-BF16-00001-of-00002.gguf \
  --llama-root /src/llama.cpp \
  --run-id muse-glimmer-q4s8-v1 \
  --output-root /models/q4s8-runs \
  --device auto \
  --q8-fraction 10
```

The model argument is the first shard for a split GGUF. The loader discovers the remaining shards. `--device auto` discovers accelerator devices; use `--device ROCm0,ROCm1,ROCm2,ROCm3` when explicit routing is required. The calibration command defaults to the pinned three-dataset recipe below. Pass dataset IDs positionally to use another compatible set, or `--calibration-dir DIR` to reuse existing `*-calibration.txt` files.

The command creates an immutable run directory containing selected corpora, GPU-generated per-source imatrices, a merged imatrix, Q4 dry-run output, the automatic Q8 ranking, tensor overrides, quantization logs, and the final GGUF. Existing run IDs are never overwritten.

For a prepared calibration-only run:

```bash
python3 scripts/build-q4s8-calibration.py \
  --model /models/model-BF16.gguf --llama-root /src/llama.cpp \
  --run-id model-calibration --prepare-only
```

For an existing merged imatrix and no calibration collection, use the lower-level builder:

```bash
scripts/build-q4s8.sh \
  --repo /src/llama.cpp \
  --model /models/model-BF16.gguf \
  --imatrix /models/imatrix-merged.gguf \
  --output /models/model-Q4S8.gguf \
  --q8-fraction 10 \
  --protect-low-bandwidth
```

Use `--plan-only` to inspect the Q4 base, ranked promotions, estimated size, and generated tensor map without writing a model.

## Default calibration recipe

The default corpus is split equally across:

- `nvidia/Open-SWE-Traces`
- `nvidia/Nemotron-SFT-Math-v4`
- `nvidia/ChatQA2-Long-SFT-data`

The pinned revisions are in [`configs/three-dataset-revisions.json`](configs/three-dataset-revisions.json). The default uses seed `20260808`, 1,000 chunks, context/batch size 512, global normalized-text deduplication, a disjoint 10% held-out split, and automatic record clipping. These defaults reproduce the earlier Qwen3.6 calibration selection, but the resulting imatrix is regenerated for each target model.

`llama-imatrix` is invoked with all discovered accelerator devices, `-ngl 999`, layer split mode, and `--process-output`. MTP-aware collection is automatic when the model exposes `nextn` tensors; use `--mtp on|off` to override detection.

## Automatic Q4/Q8 selection

`scripts/build-q4s8.sh` first runs `llama-quantize --dry-run MODEL Q4_0`. The ranker then reads the BF16 tensor values and optional imatrix `in_sum2` values, estimates Q4_0 versus Q8_0 reconstruction error, and selects promotions until the requested final Q8 byte fraction is reached. `--protect-low-bandwidth` includes small attention/state/shared/output-related candidates before the ranked fill. The generated plan is model-specific and is saved with the run; no hand-written model override is required.

The ranker is a heuristic and must be followed by quality validation. It is not a claim that every model benefits from the same Q8 fraction.

## Requirements

- BF16 GGUF source, including split GGUF support through the first shard;
- llama.cpp build with `llama-imatrix` and `llama-quantize`;
- Python packages in [`requirements.txt`](requirements.txt);
- enough storage for source, imatrices, plans, and output shards.

Quantization itself is performed by llama.cpp's quantizer; GPU acceleration applies to calibration/imatrix collection. Tensor ranking and quantization remain CPU-side in the current llama.cpp tools.

## Historical result

The repository includes the original Qwen3.6-35B-A3B Q4S8 result and measurements as historical evidence in `results/`. They document one model-specific run; the scripts are now model-agnostic. The old Qwen-named files were renamed to generic names. The MTP patch is retained for older fork-specific MTP builds.