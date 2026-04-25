"""
Milestone 1: KV Cache Compression Evaluation (H2O / ExpectedAttentionPress).

This script evaluates Qwen3-VL-4B on any supported benchmark while applying
H2O KV cache compression via the kvpress library. It records per-sample
prefill time, decode time, throughput, GPU memory, and accuracy, then saves
results to results/kv_compression/.

How H2O compression works
--------------------------
During the prefill pass, ExpectedAttentionPress accumulates the mean attention
weight each token receives across all heads. After prefill, it discards the
(compression_ratio * 100)% of tokens with the lowest cumulative attention
scores. Only the remaining "heavy-hitter" tokens are kept in the KV cache for
decoding, reducing memory bandwidth and latency without retraining the model.

Usage examples
--------------
    # RealWorldQA with 50 % compression (keep 50 % of KV cache)
    python -m eval.eval_kv_compression \\
        --dataset realworldqa \\
        --compression_ratio 0.5 \\
        --max_samples 5

    # MathVista with 30 % compression
    python -m eval.eval_kv_compression \\
        --dataset mathvista \\
        --compression_ratio 0.3 \\
        --max_samples 2

    # DocVQA with 70 % compression
    python -m eval.eval_kv_compression \\
        --dataset docvqa \\
        --compression_ratio 0.7 \\
        --max_samples 5

    # No compression (baseline parity check, compression_ratio=0.0)
    python -m eval.eval_kv_compression \\
        --dataset mmmu \\
        --compression_ratio 0.0 \\
        --max_samples 2

Supported datasets
------------------
  realworldqa  xai-org/RealworldQA           general visual reasoning
  mathvista    AI4Math/MathVista              math reasoning from images
  mmmu         MMMU/MMMU (Math subset)        multiple-choice multimodal
  docvqa       HuggingFaceM4/DocumentVQA      document understanding

Output
------
Each run appends a JSONL file to results/kv_compression/:
    results/kv_compression/<dataset>_cr<ratio>.jsonl

Each line contains:
  sample_id, dataset, compression_ratio,
  question, ground_truth, prediction,
  prefill_ms, decode_ms,
  num_tokens_generated, throughput_tok_per_sec,
  peak_gpu_mem_gb, correct
"""

import os
import json
import time
import argparse
from threading import Thread

from datasets import load_dataset
from transformers import TextIteratorStreamer

from src.load_model import load_model_and_processor
from src.utils import reset_gpu_memory, get_peak_gpu_memory_mb
from src.kv_compression import make_h2o_press


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "realworldqa": {
        "hf_name": "xai-org/RealworldQA",
        "hf_subset": None,
        "split": "test",
        "max_new_tokens": 64,
    },
    "mathvista": {
        "hf_name": "AI4Math/MathVista",
        "hf_subset": None,
        "split": "test",
        "max_new_tokens": 32,
    },
    "mmmu": {
        "hf_name": "MMMU/MMMU",
        "hf_subset": "Math",
        "split": "validation",
        "max_new_tokens": 32,
    },
    "docvqa": {
        "hf_name": "HuggingFaceM4/DocumentVQA",
        "hf_subset": None,
        "split": "validation",
        "max_new_tokens": 32,
    },
}


def load_benchmark(name: str):
    cfg = DATASET_CONFIGS[name]
    if cfg["hf_subset"]:
        return load_dataset(cfg["hf_name"], cfg["hf_subset"], split=cfg["split"])
    return load_dataset(cfg["hf_name"], split=cfg["split"])


def extract_sample(name: str, sample: dict):
    """Return (image, question_text, ground_truth_str) for a raw dataset row."""
    if name == "realworldqa":
        return sample["image"], sample["question"], str(sample["answer"])

    if name == "mathvista":
        return (
            sample.get("decoded_image"),
            sample["query"],
            str(sample.get("answer", "N/A")),
        )

    if name == "mmmu":
        options = sample["options"]
        question = (
            sample["question"]
            + "\nOptions:\n"
            + "\n".join(options)
            + "\nAnswer with only the correct option letter and nothing else."
        )
        return sample["image_1"], question, str(sample["answer"])

    if name == "docvqa":
        answers = sample.get("answers", ["N/A"])
        # Keep the full list so accuracy can be checked against any valid answer
        return sample["image"], sample["question"], answers

    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return text.strip().lower()


def is_correct(prediction: str, ground_truth) -> bool:
    """
    Approximate accuracy check.

    - Multiple-choice datasets (mmmu, realworldqa): check whether the first
      character of the prediction matches the expected answer letter.
    - DocVQA: accept any answer from the list.
    - MathVista: exact-match after stripping and lower-casing.
    """
    pred = _normalize(prediction)
    if isinstance(ground_truth, list):
        # docvqa: any correct answer counts
        return any(_normalize(gt) in pred or pred in _normalize(gt) for gt in ground_truth)
    gt = _normalize(str(ground_truth))
    # For single-letter answers (A/B/C/D), match just the first character
    if len(gt) == 1 and gt.isalpha():
        return pred[:1] == gt
    return gt in pred or pred.startswith(gt)


# ---------------------------------------------------------------------------
# Inference with split prefill / decode timing
# ---------------------------------------------------------------------------

def generate_with_timing(model, processor, image, question: str, max_new_tokens: int):
    """
    Run model.generate() in a background thread while the main thread consumes
    a TextIteratorStreamer. This lets us record the exact moment the first
    token arrives (= end of prefill) without modifying model internals.

    Returns
    -------
    prediction       : str   full decoded output
    prefill_ms       : float time from call start to first token (ms)
    decode_ms        : float time from first token to last token (ms)
    num_tokens       : int   number of generated tokens
    throughput       : float tokens per second during decode
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # skip_prompt=True so we only collect newly generated tokens
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    generate_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)

    t_start = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=generate_kwargs, daemon=True)
    thread.start()

    t_first_token = None
    chunks = []
    for chunk in streamer:
        if t_first_token is None:
            t_first_token = time.perf_counter()
        chunks.append(chunk)

    thread.join()
    t_end = time.perf_counter()

    if t_first_token is None:
        # Edge case: model generated 0 new tokens
        t_first_token = t_end

    prediction = "".join(chunks)
    prefill_ms = (t_first_token - t_start) * 1000
    decode_ms = (t_end - t_first_token) * 1000

    # Count tokens in the generated output (not the prompt)
    num_tokens = len(
        processor.tokenizer(prediction, add_special_tokens=False)["input_ids"]
    )
    decode_sec = max(t_end - t_first_token, 1e-9)
    throughput = num_tokens / decode_sec

    return prediction, prefill_ms, decode_ms, num_tokens, throughput


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-4B with H2O KV cache compression."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Benchmark to evaluate on.",
    )
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of KV cache entries to EVICT (0.0 = no compression, "
            "0.5 = evict 50%%, 0.7 = evict 70%%). Default: 0.5."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of dataset samples to evaluate. Default: 5.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build output path: e.g. results/kv_compression/realworldqa_cr0p50.jsonl
    results_dir = "results/kv_compression"
    os.makedirs(results_dir, exist_ok=True)
    ratio_tag = f"{args.compression_ratio:.2f}".replace(".", "p")
    results_path = f"{results_dir}/{args.dataset}_cr{ratio_tag}.jsonl"

    print("=" * 80)
    print("Milestone 1: KV Cache Compression (H2O / ExpectedAttentionPress)")
    print("=" * 80)
    print(f"  Dataset          : {args.dataset}")
    print(f"  Compression ratio: {args.compression_ratio}  "
          f"(evict {args.compression_ratio * 100:.0f}% of KV cache)")
    print(f"  Max samples      : {args.max_samples}")
    print(f"  Results path     : {results_path}")
    print()

    print("Loading model and processor...")
    model, processor = load_model_and_processor()

    print(f"Building H2O press (compression_ratio={args.compression_ratio})...")
    press = make_h2o_press(args.compression_ratio)

    print(f"Loading dataset: {args.dataset}...")
    dataset = load_benchmark(args.dataset)
    cfg = DATASET_CONFIGS[args.dataset]

    n_correct = 0
    n_total = 0

    with open(results_path, "w") as f:
        for idx in range(min(args.max_samples, len(dataset))):
            sample = dataset[idx]
            image, question, ground_truth = extract_sample(args.dataset, sample)

            reset_gpu_memory()

            # Apply H2O compression for this inference call only
            with press(model):
                prediction, prefill_ms, decode_ms, num_tokens, throughput = (
                    generate_with_timing(
                        model, processor, image, question, cfg["max_new_tokens"]
                    )
                )

            peak_mem_gb = get_peak_gpu_memory_mb() / 1024.0
            correct = is_correct(prediction, ground_truth)
            if correct:
                n_correct += 1
            n_total += 1

            row = {
                "sample_id": idx,
                "dataset": args.dataset,
                "compression_ratio": args.compression_ratio,
                "question": question,
                "ground_truth": ground_truth if isinstance(ground_truth, str) else ground_truth,
                "prediction": prediction,
                "prefill_ms": round(prefill_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "num_tokens_generated": num_tokens,
                "throughput_tok_per_sec": round(throughput, 2),
                "peak_gpu_mem_gb": round(peak_mem_gb, 3),
                "correct": correct,
            }

            f.write(json.dumps(row) + "\n")
            f.flush()

            # Console summary for this sample
            gt_display = (
                ground_truth if isinstance(ground_truth, str)
                else " | ".join(ground_truth[:3])
            )
            print("-" * 80)
            print(f"[{idx:>3}] Q : {question[:90]}")
            print(f"       GT: {gt_display[:60]}")
            print(f"      Pred: {prediction[:60]}")
            print(
                f"      Prefill: {prefill_ms:>7.1f} ms  "
                f"Decode: {decode_ms:>7.1f} ms  "
                f"Throughput: {throughput:>5.1f} tok/s  "
                f"Mem: {peak_mem_gb:.3f} GB  "
                f"Correct: {correct}"
            )

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    print("=" * 80)
    print(f"FINAL ACCURACY : {n_correct}/{n_total} = {accuracy:.1%}")
    print(f"Results saved to {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
