# Method

## Recipe

The reported `qwen35-q4s8-v1` run starts from BF16 and never requantizes an existing low-bit model. `llama-quantize` receives a Q4_0 default plus the exact overrides in `configs/qwen35-q4s8-v1-overrides.txt` and a merged importance matrix.

The final map has 753 tensors: 370 F32, 119 Q4_0, 10 Q4_1, and 254 Q8_0. Quantization reported `20,152.29 MiB` and `4.76 BPW`, versus `67,764.29 MiB` and `16.01 BPW` for the BF16 source.

## Calibration

The calibration pipeline allocates 1,000 chunks as `334/333/333` across Open-SWE, Nemotron Math, and ChatQA2 Long. Dataset revisions are pinned in `results/qwen35-q4s8-v1-revisions.json`.

Selection properties:

- seed: `20260808`;
- context and batch: 512;
- global normalized-text deduplication;
- stable SHA-256 sampling order;
- 10% held-out split before calibration selection;
- 24,000-character maximum record length;
- equal COT/TIR quotas for Nemotron Math;
- equal long-SFT/NarrativeQA quotas for ChatQA2;
- all 23 selected Open-SWE parquet shards are eligible.

`LLAMA_IMATRIX_PROCESS_MTP=1` creates a target context and an MTP context. Each draft token is paired with the preceding target hidden state, matching the speculative MTP path. One importance matrix is collected per source and the three matrices are merged before quantization.

## Quality evaluation

For each corpus, BF16 logits are first written as a KLD reference. The quantized model is then evaluated against the same reference:

```bash
llama-perplexity -m "$BF16" -f "$CORPUS" \
  --kl-divergence-base "$BASE_KLD" \
  -c 512 -b 512 --chunks "$CHUNKS" \
  -ngl all -sm layer -ts 1/1/1/1 -fa on

llama-perplexity -m "$Q4S8" -f "$CORPUS" \
  --kl-divergence-base "$BASE_KLD" --kl-divergence \
  -c 512 -b 512 --chunks "$CHUNKS" \
  -ngl all -sm layer -ts 1/1/1/1 -fa on
```

Reported sets were 20 code chunks, 20 Wikitext chunks, and 500 disjoint held-out Open-SWE chunks. The report includes PPL, mean KLD, and same-top-token agreement. Open-SWE KLD has a long tail, so median and upper percentiles are also disclosed.

## Throughput evaluation

Prompt processing used three repetitions:

```bash
llama-bench -m "$Q4S8" -p 4096 -n 0 -r 3 \
  -b 2048 -ub 256 -ngl 999 -sm layer -ts 1/1/1/1 \
  -fa on -ctk f16 -ctv f16 -o jsonl
```

Generation used `llama-server`, one slot, a 4,096-token context, temperature 0, seed 123, and 512 predicted tokens per prompt. The six prompt classes were GPU explanation, coding, math, factual knowledge, reasoning, and software engineering. Normal decoding and embedded-MTP decoding used the same model and prompts. MTP used `--spec-type draft-mtp --spec-draft-n-max 3`.

Hardware and runtime configuration:

- 4 × Radeon Pro V620 32 GiB;
- ROCm backend;
- layer split `1/1/1/1`;
- Flash Attention enabled;
- batch 2048, microbatch 256;
- F16 K/V cache.

The benchmark executable reported llama.cpp build commit `4dd942f9f`. These measurements predate the later gfx1030 kernel experiments and should be treated as a model-recipe result, not a claim for those kernel optimizations.

## Limitations

- KLD, PPL, and same-top agreement do not establish downstream benchmark accuracy.
- MTP throughput and acceptance vary substantially by prompt; the report gives both means and ranges.
- Results are specific to this model revision, tensor map, calibration data, and four-V620 setup.
- The MTP imatrix patch relies on fork-specific MTP APIs and is not guaranteed to apply unchanged to arbitrary llama.cpp revisions.
- No model weights, calibration text, or generated responses are stored in this repository.