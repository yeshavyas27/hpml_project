"""
Unified eval for KV cache compression methods on Qwen3-VL-4B.

Supports four methods:
  h2o        - ExpectedAttentionPress (heavy-hitter style)
  streaming  - StreamingLLMPress (attention sinks + recent window)
  snapkv     - SnapKVPress (prompt-aware, uses last window of queries)
  pyramid    - PyramidKVPress (SnapKV + pyramid budget across layers)
  modality   - ModalityAwarePress (this project's novel contribution)

This script is deliberately kept as a superset of eval_kv_compression.py so
that the H2O baseline numbers you already have remain exactly reproducible
with `--method h2o --compression_ratio 0.5`.

Usage
-----
    # H2O baseline at 50% compression on RealWorldQA, 5 samples
    python -m eval.eval_kv_methods \\
        --method h2o --dataset realworldqa --compression_ratio 0.5 --max_samples 5

    # SnapKV with 64-token probe window
    python -m eval.eval_kv_methods \\
        --method snapkv --dataset docvqa --compression_ratio 0.5 --max_samples 50

    # Modality-aware: aggressive on images, gentle on text
    python -m eval.eval_kv_methods \\
        --method modality --dataset docvqa \\
        --image_compression_ratio 0.7 --text_compression_ratio 0.2 \\
        --max_samples 50

Output
------
Results are appended as JSONL to results/kv_compression/:
    <dataset>_<method>_cr<ratio>.jsonl                  (uniform methods)
    <dataset>_modality_i<img>_t<txt>.jsonl              (modality-aware)
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
from src.kv_compression import make_press, ModalityAwarePress
from src.modality_mask import build_image_mask, summarize_mask


# ---------------------------------------------------------------------------
# Dataset configuration (identical to eval_kv_compression.py)
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
        return sample["image"], sample["question"], answers
    raise ValueError(f"Unknown dataset: {name}")


def _normalize(text: str) -> str:
    return text.strip().lower()


def is_correct(prediction: str, ground_truth) -> bool:
    pred = _normalize(prediction)
    if isinstance(ground_truth, list):
        return any(_normalize(gt) in pred or pred in _normalize(gt) for gt in ground_truth)
    gt = _normalize(str(ground_truth))
    if len(gt) == 1 and gt.isalpha():
        return pred[:1] == gt
    return gt in pred or pred.startswith(gt)


# ---------------------------------------------------------------------------
# Inference with prefill / decode timing (same as eval_kv_compression.py)
# ---------------------------------------------------------------------------

def build_inputs(processor, model, image, question: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


def generate_with_timing(model, processor, inputs, max_new_tokens: int):
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
        t_first_token = t_end

    prediction = "".join(chunks)
    prefill_ms = (t_first_token - t_start) * 1000
    decode_ms = (t_end - t_first_token) * 1000

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
    p = argparse.ArgumentParser(description="Evaluate KV cache compression methods on Qwen3-VL-4B.")
    p.add_argument("--method", required=True,
                   choices=["h2o", "streaming", "snapkv", "pyramid", "modality"])
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--max_samples", type=int, default=5)

    # Uniform methods
    p.add_argument("--compression_ratio", type=float, default=0.5,
                   help="Eviction fraction for uniform methods (h2o/streaming/snapkv/pyramid).")
    p.add_argument("--n_sink", type=int, default=4, help="StreamingLLM sink size.")
    p.add_argument("--window_size", type=int, default=64,
                   help="SnapKV / Pyramid / Modality(snapkv inner) probe window.")
    p.add_argument("--kernel_size", type=int, default=5, help="SnapKV / Pyramid pooling kernel.")
    p.add_argument("--beta", type=int, default=20, help="PyramidKV steepness.")

    # Modality-aware
    p.add_argument("--image_compression_ratio", type=float, default=0.7)
    p.add_argument("--text_compression_ratio", type=float, default=0.2)
    p.add_argument("--inner", choices=["h2o", "snapkv"], default="h2o",
                   help="Inner scorer for modality-aware press.")
    return p.parse_args()


def results_path(args) -> str:
    results_dir = "results/kv_compression"
    os.makedirs(results_dir, exist_ok=True)
    if args.method == "modality":
        i = f"{args.image_compression_ratio:.2f}".replace(".", "p")
        t = f"{args.text_compression_ratio:.2f}".replace(".", "p")
        tag = f"modality-{args.inner}_i{i}_t{t}"
    else:
        r = f"{args.compression_ratio:.2f}".replace(".", "p")
        tag = f"{args.method}_cr{r}"
    return f"{results_dir}/{args.dataset}_{tag}.jsonl"


def build_press(args):
    if args.method == "h2o":
        return make_press("h2o", compression_ratio=args.compression_ratio)
    if args.method == "streaming":
        return make_press("streaming",
                          compression_ratio=args.compression_ratio,
                          n_sink=args.n_sink)
    if args.method == "snapkv":
        return make_press("snapkv",
                          compression_ratio=args.compression_ratio,
                          window_size=args.window_size,
                          kernel_size=args.kernel_size)
    if args.method == "pyramid":
        return make_press("pyramid",
                          compression_ratio=args.compression_ratio,
                          window_size=args.window_size,
                          kernel_size=args.kernel_size,
                          beta=args.beta)
    if args.method == "modality":
        return make_press("modality",
                          image_compression_ratio=args.image_compression_ratio,
                          text_compression_ratio=args.text_compression_ratio,
                          inner=args.inner,
                          window_size=args.window_size)
    raise ValueError(args.method)


def main():
    args = parse_args()
    out_path = results_path(args)

    print("=" * 80)
    print("KV Cache Compression Eval")
    print("=" * 80)
    print(f"  Method       : {args.method}")
    print(f"  Dataset      : {args.dataset}")
    if args.method == "modality":
        print(f"  Image ratio  : {args.image_compression_ratio}")
        print(f"  Text ratio   : {args.text_compression_ratio}")
        print(f"  Inner scorer : {args.inner}")
    else:
        print(f"  Compression  : {args.compression_ratio}")
    print(f"  Max samples  : {args.max_samples}")
    print(f"  Output       : {out_path}")
    print()

    print("Loading model and processor...")
    model, processor = load_model_and_processor()

    press = build_press(args)

    print(f"Loading dataset: {args.dataset}...")
    dataset = load_benchmark(args.dataset)
    cfg = DATASET_CONFIGS[args.dataset]

    n_correct = 0
    n_total = 0

    with open(out_path, "w") as f:
        for idx in range(min(args.max_samples, len(dataset))):
            sample = dataset[idx]
            image, question, ground_truth = extract_sample(args.dataset, sample)

            # Build inputs up front so we can inspect input_ids for the modality mask.
            inputs = build_inputs(processor, model, image, question)

            mask_summary = None
            if isinstance(press, ModalityAwarePress):
                img_mask = build_image_mask(inputs["input_ids"][0], processor)
                press.set_modality_mask(img_mask)
                mask_summary = summarize_mask(img_mask)

            reset_gpu_memory()

            with press(model):
                prediction, prefill_ms, decode_ms, num_tokens, throughput = (
                    generate_with_timing(model, processor, inputs, cfg["max_new_tokens"])
                )

            peak_mem_gb = get_peak_gpu_memory_mb() / 1024.0
            correct = is_correct(prediction, ground_truth)
            if correct:
                n_correct += 1
            n_total += 1

            row = {
                "sample_id": idx,
                "dataset": args.dataset,
                "method": args.method,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "prefill_ms": round(prefill_ms, 2),
                "decode_ms": round(decode_ms, 2),
                "num_tokens_generated": num_tokens,
                "throughput_tok_per_sec": round(throughput, 2),
                "peak_gpu_mem_gb": round(peak_mem_gb, 3),
                "correct": correct,
            }
            if args.method == "modality":
                row["image_compression_ratio"] = args.image_compression_ratio
                row["text_compression_ratio"] = args.text_compression_ratio
                row["inner"] = args.inner
                if mask_summary is not None:
                    row["mask_summary"] = mask_summary
                    row["effective_compression_ratio"] = round(
                        press.effective_compression_ratio(seq_len=mask_summary["n_total"]),
                        3,
                    )
            else:
                row["compression_ratio"] = args.compression_ratio
                if args.method == "streaming":
                    row["n_sink"] = args.n_sink
                if args.method in ("snapkv", "pyramid"):
                    row["window_size"] = args.window_size
                    row["kernel_size"] = args.kernel_size
                if args.method == "pyramid":
                    row["beta"] = args.beta

            f.write(json.dumps(row) + "\n")
            f.flush()

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
                f"Tput: {throughput:>5.1f} tok/s  "
                f"Mem: {peak_mem_gb:.3f} GB  "
                f"OK: {correct}"
            )
            if mask_summary is not None:
                print(
                    f"      Mask: {mask_summary['n_image']} img / "
                    f"{mask_summary['n_text']} txt "
                    f"({mask_summary['image_fraction']*100:.1f}% image)"
                )

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    print("=" * 80)
    print(f"FINAL ACCURACY : {n_correct}/{n_total} = {accuracy:.1%}")
    print(f"Results saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
