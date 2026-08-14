# gguf-q4s8

A simple, reproducible workflow for making mixed-precision GGUF models.

Q4S8 means:

- Q4_0 is the default tensor type;
- selected tensors are automatically promoted to Q8_0;
- the selection is based on the BF16 model and calibration data;
- no new GGUF tensor type is introduced.

The workflow works with dense, MoE, MTP, single-file, and split BF16 GGUF models, provided the installed llama.cpp build supports the model.

## What you need

- a BF16 GGUF model;
- a built llama.cpp tree containing `llama-imatrix` and `llama-quantize`;
- Python 3 and the packages in [`requirements.txt`](requirements.txt);
- enough disk space for the source model, calibration files, imatrices, plans, and output model.

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Check the llama.cpp tools and available GPUs:

```bash
/path/to/llama.cpp/build/bin/llama-imatrix --list-devices
/path/to/llama.cpp/build/bin/llama-quantize --help
```

## End-to-end quantization

Use `build-q4s8-calibration.py`. It performs the complete workflow:

1. selects calibration text;
2. runs one GPU imatrix job per calibration source;
3. merges the imatrices;
4. derives the Q4_0 base map;
5. automatically selects Q8_0 promotions;
6. quantizes the BF16 model;
7. writes a manifest, plans, logs, and output model.

Example with a split model:

```bash
python3 scripts/build-q4s8-calibration.py \
  --model /models/MyModel-BF16-00001-of-00002.gguf \
  --llama-root /path/to/llama.cpp \
  --run-id mymodel-q4s8-v1 \
  --output-root /models/q4s8-runs \
  --device auto \
  --q8-fraction 10
```

Example with a single-file model:

```bash
python3 scripts/build-q4s8-calibration.py \
  --model /models/MyModel-BF16.gguf \
  --llama-root /path/to/llama.cpp \
  --run-id mymodel-q4s8-v1 \
  --output-root /models/q4s8-runs \
  --device auto \
  --q8-fraction 10
```

For explicit GPU routing:

```bash
--device ROCm0,ROCm1,ROCm2,ROCm3
```

Use the devices reported by your llama.cpp build. `--device auto` discovers accelerator devices automatically.

The first shard is the only model path you pass for a split GGUF. The remaining source shards are discovered automatically. By default the quantizer writes one merged output GGUF; pass `--keep-split` to `scripts/build-q4s8.sh` only when sharded output is explicitly wanted.

## Calibration data

With no dataset arguments, the command uses the pinned default recipe:

- `nvidia/Open-SWE-Traces`
- `nvidia/Nemotron-SFT-Math-v4`
- `nvidia/ChatQA2-Long-SFT-data`

The pinned revisions are stored in [`configs/three-dataset-revisions.json`](configs/three-dataset-revisions.json). Sampling uses a fixed seed, deterministic hashing, global text deduplication, equal source budgets, and a held-out split. The resulting imatrix is always regenerated for the target model.

To use different compatible Hugging Face datasets, pass their IDs after the script name:

```bash
python3 scripts/build-q4s8-calibration.py \
  my-org/dataset-a my-org/dataset-b \
  --model /models/MyModel-BF16.gguf \
  --llama-root /path/to/llama.cpp \
  --run-id mymodel-custom-calibration \
  --output-root /models/q4s8-runs
```

Dataset formats and field layouts vary. The default adapters support the three pinned datasets; other datasets may require an adapter in the calibration script.

To reuse calibration text that has already been selected:

```bash
python3 scripts/build-q4s8-calibration.py \
  --model /models/MyModel-BF16.gguf \
  --llama-root /path/to/llama.cpp \
  --calibration-dir /models/calibration/datasets \
  --run-id mymodel-reuse-calibration \
  --output-root /models/q4s8-runs
```

The directory must contain files named `*-calibration.txt`.

To reuse an existing merged imatrix and skip GPU calibration:

```bash
python3 scripts/build-q4s8-calibration.py \
  --model /models/MyModel-BF16.gguf \
  --llama-root /path/to/llama.cpp \
  --imatrix /models/imatrix-merged.gguf \
  --run-id mymodel-reuse-imatrix \
  --output-root /models/q4s8-runs
```

## How Q4 and Q8 are selected

The selection is automatic. You do not write a tensor override file.

The builder:

1. runs `llama-quantize --dry-run MODEL Q4_0`;
2. treats Q4_0 as the default;
3. estimates Q4_0 and Q8_0 reconstruction error from BF16 weights;
4. weights the estimate with `in_sum2` values from the imatrix when available;
5. promotes tensors until the requested Q8 byte budget is reached;
6. saves the generated tensor map in the run directory.

The main size/quality control is:

```bash
--q8-fraction 10
```

This targets approximately 10% of quantized model bytes as Q8_0. Other useful options are:

```bash
--q8-fraction 5
--q8-fraction 15
--protect-low-bandwidth
--max-tensor-mib 128
```

The ranking is a heuristic. Every new model should be validated against BF16 before the result is treated as production-ready.

## Inspect the plan without quantizing

Use the lower-level builder to create only the Q4/Q8 plan:

```bash
scripts/build-q4s8.sh \
  --repo /path/to/llama.cpp \
  --model /models/MyModel-BF16.gguf \
  --imatrix /models/imatrix-merged.gguf \
  --output /models/MyModel-Q4S8.gguf \
  --q8-fraction 10 \
  --plan-only
```

The plan directory contains:

- the Q4_0 dry-run log;
- the Q8 ranking log;
- the generated tensor override file;
- the tensor plan;
- the estimated size summary.

## Outputs

Each run is immutable: choose a new `--run-id` for a new experiment. The run directory contains:

```text
manifest.json
 datasets/
 imatrices/
 logs/
 model/
```

The manifest records the model, datasets, revisions, device selection, imatrix, Q8 budget, and output path.

## Validation

At minimum, compare the BF16 and Q4S8 models using the same input and runtime settings:

```bash
/path/to/llama.cpp/build/bin/llama-cli \
  -m /models/MyModel-Q4S8.gguf \
  -p "Give a short explanation of quantized matrix multiplication." \
  -n 128 -ngl 999
```

For quality validation, measure BF16-referenced PPL/KLD on representative and held-out text. For performance validation, use the same GPU split, context, batch, KV-cache type, and number of repetitions for both models.

The imatrix collection uses the GPU. Tensor ranking and the llama.cpp quantizer currently run on the CPU.

## Repository layout

- `scripts/build-q4s8-calibration.py` — complete calibration and quantization workflow;
- `scripts/build-q4s8.sh` — Q4/Q8 planning and quantization from an existing imatrix;
- `scripts/rank-q4s8-promotions.py` — automatic BF16 reconstruction-error ranker;
- `configs/` — shared dataset revision configuration and historical example maps;
- `docs/METHOD.md` — implementation details and limitations;
- `results/` — historical measurements, including a Qwen example.