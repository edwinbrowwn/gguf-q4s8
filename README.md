# gguf-q4s8

A reproducible mixed-precision GGUF recipe for **Qwen3.6-35B-A3B with embedded MTP**. The project name “Q4S8” is shorthand for a Q4_0 base with selected tensors promoted to Q8_0 and sensitive state/norm tensors retained as F32. It does not introduce a new GGUF tensor type.

## Result

The selected three-dataset recipe produced a **20,152.29 MiB (4.76 BPW)** model from a 67,764.29 MiB BF16 source, a **70.26% size reduction**. The final tensor map contained:

| Type | Tensors |
|---|---:|
| F32 | 370 |
| Q4_0 | 119 |
| Q4_1 | 10 |
| Q8_0 | 254 |

The large MoE expert matrices remain predominantly Q4. Smaller attention, shared-expert, recurrent-output, and other error-sensitive tensors are promoted to Q8. Recurrent controls and normalization tensors remain F32. The output head and the draft-only MTP `eh_proj` use Q4_0.

### Quality against BF16

KLD was measured from the quantized token distribution to the BF16 reference distribution. “Same top” is the fraction of positions where both distributions selected the same highest-probability token.

| Evaluation set | BF16 PPL | Q4S8 PPL | PPL change | Mean KLD | Same top |
|---|---:|---:|---:|---:|---:|
| Code, 20 chunks | 2.571697 | 2.613995 | +1.645% | 0.023788 ± 0.000964 | 95.471% |
| Wikitext, 20 chunks | 6.597632 | 6.716739 | +1.805% | 0.027803 ± 0.000888 | 92.078% |
| Held-out Open-SWE, 500 chunks | 2.179704 | 2.198697 | +0.871% | 0.108394 ± 0.001881 | 93.684% |

The held-out Open-SWE KLD is heavy-tailed: median `0.001331`, 95th percentile `0.311899`, and 99th percentile `2.478030`. KLD and same-top agreement measure distribution fidelity; they are not substitutes for downstream task evaluation.

### Performance

Measured on four Radeon Pro V620 32 GiB GPUs with ROCm, layer split `1/1/1/1`, Flash Attention enabled, `-b 2048`, `-ub 256`, and F16 KV cache.

| Test | Result |
|---|---:|
| PP4096 | **6246.05 tok/s** |
| Normal decode, six-prompt mean | **81.393 tok/s** |
| MTP decode, six-prompt mean | **120.398 tok/s** |
| MTP draft acceptance | **64.012%** |
| MTP throughput gain over normal decode | **+47.92%** |

PP4096 samples were `6245.76`, `6251.07`, and `6241.30` tok/s. Across the six generation prompts, normal decode ranged from `81.22–81.55` tok/s; MTP ranged from `107.87–147.27` tok/s with acceptance from `53.66–87.03%`.

A same-recipe, five-prompt A/B test of the draft-only MTP projection found Q4_0 effectively neutral relative to Q8_0: `119.288` vs `118.623` tok/s and `62.902%` vs `62.881%` mean acceptance, while saving about 4 MiB.

Machine-readable measurements are in [`results/qwen35-q4s8-v1.json`](results/qwen35-q4s8-v1.json).

## Calibration and quantization

The reported model used 1,000 calibration chunks split equally across:

- `nvidia/Open-SWE-Traces`
- `nvidia/Nemotron-SFT-Math-v4`
- `nvidia/ChatQA2-Long-SFT-data`

Dataset revisions are pinned in [`results/qwen35-q4s8-v1-revisions.json`](results/qwen35-q4s8-v1-revisions.json). Sampling uses a fixed seed (`20260808`), stable SHA-256 ordering, global deduplication, a disjoint 10% held-out split, and a 24,000-character record cap. Per-source MTP-aware importance matrices are collected and then merged before quantization.

The exact tensor map is [`configs/qwen35-q4s8-v1-overrides.txt`](configs/qwen35-q4s8-v1-overrides.txt).

## Reproduce

Requirements:

1. A BF16 Qwen3.6-35B-A3B GGUF with embedded MTP.
2. A built, MTP-capable llama.cpp tree.
3. Python packages from `requirements.txt`.
4. About 70 GiB for the BF16 model plus output and calibration artifacts.

The calibration pipeline needs an MTP-aware `llama-imatrix`. If the target llama.cpp tree does not contain that probe, apply [`patches/llama-imatrix-mtp.patch`](patches/llama-imatrix-mtp.patch). The patch depends on the MTP context/embedding APIs from the associated llama.cpp fork and may require rebasing against other revisions.

```bash
git clone https://github.com/edwinbrowwn/gguf-q4s8.git
cd gguf-q4s8
python3 -m pip install -r requirements.txt

python3 scripts/build-qwen35-three-dataset-mtp-recipe.py \
  nvidia/Open-SWE-Traces \
  nvidia/Nemotron-SFT-Math-v4 \
  nvidia/ChatQA2-Long-SFT-data \
  --model /models/Qwen3.6-35B-A3B-MTP-BF16.gguf \
  --llama-root /src/llama.cpp \
  --revisions-file results/qwen35-q4s8-v1-revisions.json \
  --run-id qwen35-q4s8-v1 \
  --batch-size 512 \
  --context-size 512 \
  --iterations 1000 \
  --seed 20260808 \
  --holdout-fraction 0.10 \
  --download-workers 3 \
  --candidate-records 4000 \
  --max-record-chars 24000
```

For type-map experiments without the three-dataset pipeline, use:

```bash
scripts/build-qwen35-q4-0-s8.sh \
  --repo /src/llama.cpp \
  --input /models/Qwen3.6-35B-A3B \
  --out-dir /models/qwen35-q4s8 \
  --stage auto \
  --auto-q8-fraction 10 \
  --threads 24
```

See [`docs/METHOD.md`](docs/METHOD.md) for measurement commands and limitations.

## Repository contents

- `configs/`: exact reported tensor-type policy.
- `scripts/build-qwen35-three-dataset-mtp-recipe.py`: deterministic calibration, MTP imatrix collection, merge, and quantization.
- `scripts/build-qwen35-q4-0-s8.sh`: stock/fixed/native/automatic tensor-map builder.
- `scripts/rank-q4-0-q8-promotions.py`: BF16 reconstruction-error ranker.
- `scripts/optimize-qwen35-s8.sh`: candidate KLD and throughput sweep.
- `patches/`: MTP-aware imatrix integration used by the calibration pipeline.
- `results/`: concise, machine-readable evidence for the reported run.