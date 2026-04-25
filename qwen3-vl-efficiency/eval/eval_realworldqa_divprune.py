"""
eval_realworldqa_divprune.py — RealWorldQA evaluation with DivPrune visual token pruning.

Usage:
    python -m eval.eval_realworldqa_divprune --subset_ratio 0.5 --max_samples 100
    python -m eval.eval_realworldqa_divprune --no_divprune --max_samples 100
"""

import os
import json
import argparse
from threading import Thread

from datasets import load_dataset
from transformers import TextIteratorStreamer

from src.load_model import load_model_and_processor
from src.utils import reset_gpu_memory, get_peak_gpu_memory_mb
from src.divprune import (
    DivPruneConfig, apply_divprune_to_qwen,
    get_divprune_stats, is_divprune_active,
)

import time

DATASET_HF = "xai-org/RealworldQA"
SPLIT = "test"
MAX_NEW_TOKENS = 64


def _normalize(text: str) -> str:
    return text.strip().lower()


def is_correct(prediction: str, ground_truth: str) -> bool:
    pred = _normalize(prediction)
    gt = _normalize(str(ground_truth))
    if len(gt) == 1 and gt.isalpha():
        return pred[:1] == gt
    return gt in pred or pred.startswith(gt)


def generate_with_timing(model, processor, image, question: str):
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

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    generate_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS)

    t_start = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=generate_kwargs, daemon=True)
    thread.start()

    t_first = None
    chunks = []
    for chunk in streamer:
        if t_first is None:
            t_first = time.perf_counter()
        chunks.append(chunk)

    thread.join()
    t_end = time.perf_counter()

    if t_first is None:
        t_first = t_end

    prediction = "".join(chunks)
    prefill_ms = (t_first - t_start) * 1000
    decode_ms = (t_end - t_first) * 1000
    num_tokens = len(processor.tokenizer(prediction, add_special_tokens=False)["input_ids"])
    throughput = num_tokens / max(t_end - t_first, 1e-9)

    return prediction, prefill_ms, decode_ms, num_tokens, throughput


def parse_args():
    parser = argparse.ArgumentParser(description="RealWorldQA evaluation with DivPrune.")
    parser.add_argument("--subset_ratio", type=float, default=0.5)
    parser.add_argument("--no_divprune", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    tag = "baseline" if args.no_divprune else f"r{args.subset_ratio:.2f}".replace(".", "p")
    results_dir = "results/divprune"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/realworldqa_{tag}.jsonl"

    print("=" * 80)
    print("RealWorldQA Evaluation — DivPrune")
    print("=" * 80)
    print(f"  DivPrune       : {'disabled (baseline)' if args.no_divprune else 'enabled'}")
    if not args.no_divprune:
        print(f"  Subset ratio   : {args.subset_ratio}")
    print(f"  Results path   : {results_path}")
    print()

    print("Loading model...")
    model, processor = load_model_and_processor()

    if not args.no_divprune:
        config = DivPruneConfig(
            use_divprune=True,
            divprune_r=args.subset_ratio,
            verbose=args.verbose,
        )
        apply_divprune_to_qwen(model, config)

    print(f"Loading dataset: {DATASET_HF} / {SPLIT}...")
    dataset = load_dataset(DATASET_HF, split=SPLIT)

    n_correct = 0
    n_total = 0
    limit = args.max_samples if args.max_samples else len(dataset)

    with open(results_path, "w") as f:
        for idx in range(min(limit, len(dataset))):
            sample = dataset[idx]

            image = sample["image"]
            question = sample["question"]
            ground_truth = str(sample["answer"])

            reset_gpu_memory()

            prediction, prefill_ms, decode_ms, num_tokens, throughput = generate_with_timing(
                model, processor, image, question
            )

            peak_mem_gb = get_peak_gpu_memory_mb() / 1024.0
            correct = is_correct(prediction, ground_truth)
            if correct:
                n_correct += 1
            n_total += 1

            divprune_info = {}
            if is_divprune_active(model):
                stats = get_divprune_stats(model)
                divprune_info = {
                    "pruning_applied": stats.pruning_applied,
                    "total_image_tokens": stats.total_image_tokens,
                    "kept_image_tokens": stats.kept_image_tokens,
                    "original_seq_len": stats.original_seq_len,
                    "pruned_seq_len": stats.pruned_seq_len,
                }

            row = {
                "sample_id": idx,
                "dataset": "realworldqa",
                "subset_ratio": args.subset_ratio if not args.no_divprune else None,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "prefill_ms": round(prefill_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "num_tokens_generated": num_tokens,
                "throughput_tok_per_sec": round(throughput, 2),
                "peak_gpu_mem_gb": round(peak_mem_gb, 3),
                "correct": correct,
                "divprune": divprune_info,
            }

            f.write(json.dumps(row) + "\n")
            f.flush()

            print(
                f"[{idx:>4}] GT={ground_truth[:20]:<20}  Pred={prediction[:20]:<20}  "
                f"Prefill={prefill_ms:>7.1f}ms  Decode={decode_ms:>7.1f}ms  "
                f"Tput={throughput:>5.1f}tok/s  Mem={peak_mem_gb:.3f}GB  "
                f"{'✓' if correct else '✗'}"
            )

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    print("=" * 80)
    print(f"Accuracy : {n_correct}/{n_total} = {accuracy:.1%}")
    print(f"Results  : {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
