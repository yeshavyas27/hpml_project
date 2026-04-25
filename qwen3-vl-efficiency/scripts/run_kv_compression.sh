#!/usr/bin/env bash
# Milestone 1 launcher: KV cache compression with H2O on one dataset.
#
# Defaults (overridable via env vars):
#   DATASET      dataset name              default: realworldqa
#   RATIO        compression ratio         default: 0.5
#   MAX_SAMPLES  number of samples         default: 5
#   METHOD       h2o|streaming|snapkv|pyramid|modality   default: h2o
#
# Examples:
#   bash scripts/run_kv_compression.sh
#   METHOD=snapkv RATIO=0.5 MAX_SAMPLES=50 bash scripts/run_kv_compression.sh
#   DATASET=docvqa RATIO=0.7 bash scripts/run_kv_compression.sh
set -euo pipefail

DATASET="${DATASET:-realworldqa}"
RATIO="${RATIO:-0.5}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
METHOD="${METHOD:-h2o}"

echo "Running ${METHOD} @ ratio=${RATIO} on ${DATASET} (${MAX_SAMPLES} samples)"

python -m eval.eval_kv_methods \
    --method "${METHOD}" \
    --dataset "${DATASET}" \
    --compression_ratio "${RATIO}" \
    --max_samples "${MAX_SAMPLES}"
