#!/usr/bin/env bash
# Build a model-agnostic mixed Q4_0/Q8_0 GGUF from a BF16 GGUF source.
# The Q8_0 tensor set is selected automatically from BF16 reconstruction error
# and an optional imatrix; no handwritten model-specific map is required.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/build-q4s8.sh --model FIRST-BF16-SHARD.gguf --output MODEL-Q4S8.gguf [options]

Options:
  --repo PATH                 llama.cpp source tree (default: $LLAMA_CPP_ROOT or ~/llama.cpp)
  --build-dir PATH            build directory (default: REPO/build)
  --python PATH               Python interpreter (default: python3)
  --imatrix PATH              merged importance matrix (optional)
  --q8-fraction PCT           maximum Q8_0 fraction of quantized bytes (default: 10)
  --max-tensor-mib N          rank only tensors no larger than N MiB (default: unlimited)
  --protect-low-bandwidth     retain small attention/state/shared tensors as Q8 candidates
  --keep-split                opt in to output shards matching a split source
  --threads N                 quantizer threads (default: physical core count)
  --plan-only                 create dry-run, ranking, and override artifacts only
  --force                     replace existing plan/output artifacts
  -h, --help                  show this help

The input may be a single BF16 GGUF or the first shard of a split BF16 GGUF.
For split input, output is written as matching numbered GGUF shards.
EOF
}

die() { echo "error: $*" >&2; exit 1; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOL_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO=${LLAMA_CPP_ROOT:-$HOME/llama.cpp}
BUILD_DIR=
PYTHON=python3
MODEL=
OUTPUT=
IMATRIX=
Q8_FRACTION=10
MAX_TENSOR_MIB=0
PROTECT=0
KEEP_SPLIT=0
THREADS=
PLAN_ONLY=0
FORCE=0

while (($#)); do
    case "$1" in
        --repo) [[ $# -ge 2 ]] || die "--repo needs a path"; REPO=$2; shift 2 ;;
        --build-dir) [[ $# -ge 2 ]] || die "--build-dir needs a path"; BUILD_DIR=$2; shift 2 ;;
        --python) [[ $# -ge 2 ]] || die "--python needs a path"; PYTHON=$2; shift 2 ;;
        --model|--input) [[ $# -ge 2 ]] || die "--model needs a path"; MODEL=$2; shift 2 ;;
        --output) [[ $# -ge 2 ]] || die "--output needs a path"; OUTPUT=$2; shift 2 ;;
        --imatrix) [[ $# -ge 2 ]] || die "--imatrix needs a path"; IMATRIX=$2; shift 2 ;;
        --q8-fraction) [[ $# -ge 2 ]] || die "--q8-fraction needs a percentage"; Q8_FRACTION=$2; shift 2 ;;
        --max-tensor-mib) [[ $# -ge 2 ]] || die "--max-tensor-mib needs a size"; MAX_TENSOR_MIB=$2; shift 2 ;;
        --protect-low-bandwidth) PROTECT=1; shift ;;
        --keep-split) KEEP_SPLIT=1; shift ;;
        --threads) [[ $# -ge 2 ]] || die "--threads needs a number"; THREADS=$2; shift 2 ;;
        --plan-only) PLAN_ONLY=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (use --help)" ;;
    esac
done

[[ -n "$MODEL" ]] || { usage >&2; die "--model is required"; }
[[ -n "$OUTPUT" ]] || { usage >&2; die "--output is required"; }
REPO=$(readlink -f -- "$REPO")
[[ -n "$BUILD_DIR" ]] || BUILD_DIR="$REPO/build"
BUILD_DIR=$(readlink -f -- "$BUILD_DIR")
MODEL=$(readlink -f -- "$MODEL")
OUTPUT=$(readlink -m -- "$OUTPUT")
[[ -f "$MODEL" ]] || die "model does not exist: $MODEL"
[[ -d "$REPO" ]] || die "llama.cpp repo does not exist: $REPO"
QUANT="$BUILD_DIR/bin/llama-quantize"
[[ -x "$QUANT" ]] || die "missing quantizer: $QUANT"
RANKER="$TOOL_ROOT/scripts/rank-q4s8-promotions.py"
[[ -f "$RANKER" ]] || die "missing ranker: $RANKER"
[[ "$Q8_FRACTION" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid --q8-fraction: $Q8_FRACTION"
[[ "$MAX_TENSOR_MIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid --max-tensor-mib: $MAX_TENSOR_MIB"
awk "BEGIN { exit !($Q8_FRACTION >= 0 && $Q8_FRACTION <= 100) }" || die "Q8 fraction must be between 0 and 100"
if [[ -n "$IMATRIX" ]]; then
    IMATRIX=$(readlink -f -- "$IMATRIX")
    [[ -f "$IMATRIX" ]] || die "imatrix does not exist: $IMATRIX"
fi
if [[ -z "$THREADS" ]]; then
    THREADS=$(lscpu -p=Core 2>/dev/null | awk '!/^#/ && NF { print $1 }' | sort -u | wc -l)
    (( THREADS > 0 )) || THREADS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
fi
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "invalid thread count: $THREADS"

OUT_DIR=$(dirname -- "$OUTPUT")
OUT_STEM=$(basename -- "$OUTPUT")
OUT_STEM=${OUT_STEM%.gguf}
PLAN="$OUT_DIR/${OUT_STEM}.q4s8-plan"
BASE_LOG="$PLAN/q4-0-dry-run.log"
RANK_LOG="$PLAN/q8-ranking.log"
OVERRIDES="$PLAN/tensor-type-overrides.txt"
SUMMARY="$PLAN/summary.txt"
PLAN_TSV="$PLAN/tensor-plan.tsv"
mkdir -p -- "$PLAN"
for path in "$BASE_LOG" "$RANK_LOG" "$OVERRIDES" "$SUMMARY" "$PLAN_TSV"; do
    if [[ -e "$path" && "$FORCE" -ne 1 ]]; then
        die "artifact exists: $path (use --force)"
    fi
done

# llama-quantize accepts the first shard and discovers the complete split model.
echo "[1/4] deriving Q4_0 base map"
"$QUANT" --dry-run "$MODEL" Q4_0 >"$BASE_LOG" 2>&1

rank_args=(--bf16 "$MODEL" --base-log "$BASE_LOG" --q8-fraction "$Q8_FRACTION" --output "$OVERRIDES" --plan "$PLAN_TSV" --summary "$SUMMARY")
[[ -n "$IMATRIX" ]] && rank_args+=(--imatrix "$IMATRIX")
[[ "$MAX_TENSOR_MIB" != 0 ]] && rank_args+=(--max-tensor-mib "$MAX_TENSOR_MIB")
[[ "$PROTECT" -eq 1 ]] && rank_args+=(--protect-low-bandwidth)
echo "[2/4] selecting Q8_0 promotions automatically"
PYTHONPATH="$REPO/gguf-py${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$RANKER" "${rank_args[@]}" >"$RANK_LOG" 2>&1
cat "$RANK_LOG"
cat "$SUMMARY"

if [[ "$PLAN_ONLY" -eq 1 ]]; then
    echo "plan-only requested; no model written"
    exit 0
fi

if [[ -e "$OUTPUT" && "$FORCE" -ne 1 ]]; then
    die "output exists: $OUTPUT (use --force)"
fi

echo "[3/4] quantizing with automatic tensor map"
# A split BF16 source is read as a whole, but the default deliverable is one
# merged GGUF. Sharding is explicit opt-in so a two-shard source cannot silently
# produce a two-or-more-shard final model.
quant_args=(--pure --tensor-type-file "$OVERRIDES")
[[ "$KEEP_SPLIT" -eq 1 ]] && quant_args+=(--keep-split)
[[ -n "$IMATRIX" ]] && quant_args+=(--imatrix "$IMATRIX")
"$QUANT" "${quant_args[@]}" "$MODEL" "$OUTPUT" Q4_0 "$THREADS" >"$PLAN/quantize.log" 2>&1
cat "$PLAN/quantize.log"

echo "[4/4] validating output artifacts"
if [[ "$KEEP_SPLIT" -eq 1 ]]; then
    if [[ "$MODEL" =~ -[0-9]{5}-of-([0-9]{5})\.gguf$ ]]; then
        count=${BASH_REMATCH[1]}
        out_base=${OUTPUT%.gguf}
        for ((i=1; i<=10#$count; i++)); do
            shard=$(printf '%s-%05d-of-%s.gguf' "$out_base" "$i" "$count")
            [[ -s "$shard" ]] || die "missing output shard: $shard"
        done
        echo "validated $count split output shards"
    else
        die "--keep-split requires a split source"
    fi
else
    [[ -s "$OUTPUT" ]] || die "missing merged output: $OUTPUT"
    # Guard the single-file contract: no numbered output shard may be left beside
    # the requested file, even if a stale artifact existed from an older run.
    if compgen -G "${OUTPUT%.gguf}-?????-of-?????.gguf" >/dev/null; then
        die "numbered output shards found beside single-file output"
    fi
    echo "validated single merged output: $OUTPUT"
fi

echo "Q4S8 build complete"
echo "  output:   $OUTPUT"
echo "  plan:     $PLAN"
echo "  overrides:$OVERRIDES"