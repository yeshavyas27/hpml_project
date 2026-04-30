#!/usr/bin/env python3
"""Summarize DivPrune benchmark results: compute per-file averages across all samples."""

import json
import os
import glob
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "divprune_results")

NUMERIC_FIELDS = [
    "prefill_ms",
    "decode_ms",
    "num_tokens_generated",
    "throughput_tok_per_sec",
    "peak_gpu_mem_gb",
]

DIVPRUNE_FIELDS = [
    "total_image_tokens",
    "kept_image_tokens",
    "original_seq_len",
    "pruned_seq_len",
]


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(records):
    n = len(records)
    if n == 0:
        return {}

    sums = defaultdict(float)
    counts = defaultdict(int)

    for r in records:
        # accuracy
        if "correct" in r:
            sums["accuracy"] += float(r["correct"])
            counts["accuracy"] += 1

        for field in NUMERIC_FIELDS:
            if field in r and r[field] is not None:
                sums[field] += r[field]
                counts[field] += 1

        dp = r.get("divprune", {})
        for field in DIVPRUNE_FIELDS:
            if field in dp and dp[field] is not None:
                sums[f"divprune.{field}"] += dp[field]
                counts[f"divprune.{field}"] += 1

    result = {"num_samples": n}
    for key in sums:
        result[key] = sums[key] / counts[key] if counts[key] else None

    # token retention ratio
    if "divprune.kept_image_tokens" in result and "divprune.total_image_tokens" in result:
        kept = result["divprune.kept_image_tokens"]
        total = result["divprune.total_image_tokens"]
        result["divprune.retention_ratio"] = kept / total if total else None

    # seq len reduction ratio
    if "divprune.pruned_seq_len" in result and "divprune.original_seq_len" in result:
        pruned = result["divprune.pruned_seq_len"]
        original = result["divprune.original_seq_len"]
        result["divprune.seq_reduction_ratio"] = pruned / original if original else None

    return result


def fmt(v):
    if v is None:
        return "  N/A  "
    if isinstance(v, float):
        return f"{v:>8.4f}"
    return f"{v:>8}"


def print_table(rows):
    # Collect all metric keys (excluding dataset/ratio/num_samples for column headers)
    metric_keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in ("dataset", "subset_ratio", "num_samples") and k not in seen:
                metric_keys.append(k)
                seen.add(k)

    header_fields = ["dataset", "ratio", "n"] + metric_keys
    # Print header
    col_widths = {
        "dataset": 12,
        "ratio": 8,
        "n": 6,
    }
    for k in metric_keys:
        col_widths[k] = max(len(k) + 2, 10)

    def pad(s, width):
        return str(s).ljust(width)

    header = "  ".join(pad(f, col_widths.get(f, 10)) for f in header_fields)
    print(header)
    print("-" * len(header))

    for row in rows:
        ratio = row["subset_ratio"] if row["subset_ratio"] is not None else "baseline"
        line_parts = [
            pad(row["dataset"], col_widths["dataset"]),
            pad(ratio, col_widths["ratio"]),
            pad(row["num_samples"], col_widths["n"]),
        ]
        for k in metric_keys:
            v = row.get(k)
            if v is None:
                s = "N/A"
            elif isinstance(v, float):
                s = f"{v:.4f}"
            else:
                s = str(v)
            line_parts.append(pad(s, col_widths[k]))
        print("  ".join(line_parts))


def main():
    pattern = os.path.join(RESULTS_DIR, "*.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No JSONL files found in {RESULTS_DIR}")
        return

    # Skip duplicate "(1)" files
    files = [f for f in files if " (1)" not in os.path.basename(f)]

    rows = []
    for path in files:
        fname = os.path.basename(path)
        records = load_jsonl(path)
        if not records:
            continue

        # Infer dataset and ratio from filename (e.g. mathvista_r0p50.jsonl)
        first = records[0]
        dataset = first.get("dataset", fname.split("_")[0])
        subset_ratio = first.get("subset_ratio")  # None means baseline

        stats = summarize(records)
        row = {"dataset": dataset, "subset_ratio": subset_ratio, **stats}
        rows.append(row)

    # Sort by dataset, then ratio (None first)
    def sort_key(r):
        ratio = r["subset_ratio"] if r["subset_ratio"] is not None else -1.0
        return (r["dataset"], ratio)

    rows.sort(key=sort_key)

    print(f"\nDivPrune Results Summary — {len(rows)} files, averaged over all samples\n")
    print_table(rows)
    print()


if __name__ == "__main__":
    main()
