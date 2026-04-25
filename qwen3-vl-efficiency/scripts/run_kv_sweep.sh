#!/usr/bin/env bash
# Full sweep: every method x every dataset x a set of compression ratios,
# plus the modality-aware grid.  Produces the JSONL files you'll aggregate
# into the results table for the paper.
#
# Usage:
#   bash scripts/run_kv_sweep.sh                          # small smoke run
#   MAX_SAMPLES=50 bash scripts/run_kv_sweep.sh           # real eval run
#   DATASETS="docvqa" MAX_SAMPLES=50 bash scripts/run_kv_sweep.sh
#
# Env vars:
#   DATASETS     space-separated list, default: "realworldqa mathvista mmmu docvqa"
#   RATIOS       space-separated list, default: "0.0 0.3 0.5 0.7"
#   METHODS      space-separated list, default: "h2o streaming snapkv pyramid"
#   MAX_SAMPLES  samples per (method, dataset, ratio) cell; default 5 (smoke)
set -euo pipefail

DATASETS="${DATASETS:-realworldqa mathvista mmmu docvqa}"
RATIOS="${RATIOS:-0.0 0.3 0.5 0.7}"
METHODS="${METHODS:-h2o streaming snapkv pyramid}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"

echo "Sweep config"
echo "  datasets    : ${DATASETS}"
echo "  ratios      : ${RATIOS}"
echo "  methods     : ${METHODS}"
echo "  max_samples : ${MAX_SAMPLES}"
echo

# Uniform methods.
for dataset in ${DATASETS}; do
    for method in ${METHODS}; do
        for ratio in ${RATIOS}; do
            echo ">>> ${method} @ ratio=${ratio} on ${dataset}"
            python -m eval.eval_kv_methods \
                --method "${method}" \
                --dataset "${dataset}" \
                --compression_ratio "${ratio}" \
                --max_samples "${MAX_SAMPLES}" \
                || echo "    (failed, continuing)"
        done
    done
done

# Modality-aware grid: image / text ratio pairs.
# Rule of thumb: image >= text since images are less sensitive.
MODALITY_PAIRS=(
    "0.3 0.1"
    "0.5 0.2"
    "0.7 0.2"
    "0.7 0.3"
    "0.8 0.3"
)
for dataset in ${DATASETS}; do
    for pair in "${MODALITY_PAIRS[@]}"; do
        read -r img txt <<<"${pair}"
        echo ">>> modality @ image=${img} text=${txt} on ${dataset}"
        python -m eval.eval_kv_methods \
            --method modality \
            --dataset "${dataset}" \
            --image_compression_ratio "${img}" \
            --text_compression_ratio "${txt}" \
            --inner h2o \
            --max_samples "${MAX_SAMPLES}" \
            || echo "    (failed, continuing)"
    done
done

echo
echo "Sweep complete. Results in results/kv_compression/"
