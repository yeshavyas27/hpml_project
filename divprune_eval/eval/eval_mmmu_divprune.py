"""
eval_mmmu_divprune.py — MMMU evaluation with DivPrune visual token pruning.

DivPrune greedily selects diverse visual tokens before the first LLM layer,
keeping only subset_ratio * N_image_tokens tokens based on max-min cosine distance.

Metrics reported:
  prefill_ms, decode_ms, throughput_tok_per_sec, peak_gpu_mem_gb, correct
  plus comprehensive inference metrics matching fastv_eval_mmmu.py

Usage:
    python -m eval.eval_mmmu_divprune --subset_ratio 0.5 --max_samples 1000
    python -m eval.eval_mmmu_divprune --no_divprune --max_samples 1000   # baseline
    python -m eval.eval_mmmu_divprune --sweep_ratios 0.2 0.3 0.5 --max_samples 1000
"""

import os
import gc
import json
import re
import time
import argparse
import math
import threading
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from src.divprune import (
    DivPruneConfig, apply_divprune_to_qwen, remove_divprune,
    get_divprune_stats, is_divprune_active, get_divprune_config,
)

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False
    print("[metrics] pynvml not found – skipping hardware counters")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-VL-4B-Instruct"
DATASET_ID  = "MMMU/MMMU"
SPLIT       = "validation"
RESULTS_DIR = Path("results/divprune")

ALL_SUBJECTS = [
    "Accounting","Agriculture","Architecture_and_Engineering","Art","Art_Theory",
    "Basic_Medical_Science","Biology","Chemistry","Clinical_Medicine","Computer_Science",
    "Design","Diagnostics_and_Laboratory_Medicine","Economics","Electronics",
    "Energy_and_Power","Finance","Geography","History","Literature","Manage",
    "Marketing","Materials","Math","Mechanical_Engineering","Music","Pharmacy",
    "Physics","Psychology","Public_Health","Sociology",
]

# ── GPU constants ──────────────────────────────────────────────────────────────
GPU_PEAK_TFLOPS  = None
GPU_PEAK_BW_GBPS = None
DTYPE_BYTES      = 2  # bf16

def _detect_gpu_specs():
    global GPU_PEAK_TFLOPS, GPU_PEAK_BW_GBPS
    if not torch.cuda.is_available():
        return
    name = torch.cuda.get_device_name(0).upper()
    specs = {
        "A100": (312, 2000), "H100": (989, 3350),
        "A10":  (125, 600),  "A10G": (125, 600),
        "3090": (35.6, 936), "4090": (82.6, 1008),
        "3080": (29.8, 760), "4080": (49.0, 716),
        "T4":   (65, 300),   "V100": (125, 900),
    }
    for key, (tflops, bw) in specs.items():
        if key in name:
            GPU_PEAK_TFLOPS  = tflops * 1e12
            GPU_PEAK_BW_GBPS = bw * 1e9
            print(f"[metrics] Detected GPU: {name}  |  peak {tflops} TFLOPS  |  {bw} GB/s BW")
            return
    if HAS_PYNVML:
        GPU_PEAK_BW_GBPS = 900e9
        GPU_PEAK_TFLOPS  = 100e12
        print(f"[metrics] Unknown GPU; using conservative defaults")

_detect_gpu_specs()

# ── Dataclasses ────────────────────────────────────────────────────────────────
@dataclass
class SampleMetrics:
    image_preproc_ms    : float = 0.0
    tokenize_ms         : float = 0.0
    ttft_ms             : float = 0.0
    prefill_ms          : float = 0.0
    decode_ms           : float = 0.0
    e2e_ms              : float = 0.0
    vision_encoder_ms   : float = 0.0
    llm_ms              : float = 0.0
    input_tokens        : int   = 0
    image_tokens        : int   = 0
    output_tokens       : int   = 0
    peak_vram_bytes     : int   = 0
    kv_cache_est_bytes  : int   = 0
    prefill_flops       : float = 0.0
    mfu                 : float = 0.0
    arithmetic_intensity: float = 0.0
    prefill_tok_per_s   : float = 0.0
    decode_tok_per_s    : float = 0.0
    avg_sm_util_pct     : float = 0.0
    avg_mem_bw_util_pct : float = 0.0


@dataclass
class AggregatedMetrics:
    n_samples            : int   = 0
    mean_e2e_ms          : float = 0.0
    mean_ttft_ms         : float = 0.0
    mean_tpot_ms         : float = 0.0
    mean_prefill_ms      : float = 0.0
    mean_decode_ms       : float = 0.0
    mean_prefill_tok_s   : float = 0.0
    mean_decode_tok_s    : float = 0.0
    mean_image_tokens    : float = 0.0
    mean_input_tokens    : float = 0.0
    mean_output_tokens   : float = 0.0
    mean_peak_vram_mb    : float = 0.0
    mean_kvcache_mb      : float = 0.0
    mean_mfu_pct         : float = 0.0
    mean_arith_intensity : float = 0.0
    mean_sm_util_pct     : float = 0.0
    mean_mem_bw_util_pct : float = 0.0
    p95_e2e_ms           : float = 0.0
    p95_ttft_ms          : float = 0.0
    p95_decode_tok_s     : float = 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────
def estimate_kv_cache_bytes(model_cfg, seq_len: int) -> int:
    try:
        text_cfg = getattr(model_cfg, 'text_config', model_cfg)
        n_layers = text_cfg.num_hidden_layers
        n_kv_heads = getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)
        head_dim = getattr(text_cfg, 'head_dim', text_cfg.hidden_size // text_cfg.num_attention_heads)
        return 2 * n_layers * n_kv_heads * head_dim * seq_len * DTYPE_BYTES
    except Exception:
        return 0


def estimate_prefill_flops(model_cfg, seq_len: int) -> float:
    text_cfg = getattr(model_cfg, "text_config", model_cfg)
    layers = text_cfg.num_hidden_layers
    hidden_dim = text_cfg.hidden_size
    attn_flops = 2 * layers * (seq_len ** 2) * hidden_dim
    mlp_flops  = 4 * layers * seq_len * (hidden_dim ** 2)
    return attn_flops + mlp_flops


class HWSampler:
    """Background thread that polls pynvml every 50 ms."""
    def __init__(self, device_idx: int = 0):
        self.device_idx = device_idx
        self._sm_samples : list = []
        self._bw_samples : list = []
        self._running    = False
        self._thread     = None

    def start(self):
        if not HAS_PYNVML:
            return
        self._sm_samples.clear(); self._bw_samples.clear()
        self._running = True
        self._thread  = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> tuple:
        if not HAS_PYNVML:
            return 0.0, 0.0
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        sm = sum(self._sm_samples) / len(self._sm_samples) if self._sm_samples else 0.0
        bw = sum(self._bw_samples) / len(self._bw_samples) if self._bw_samples else 0.0
        return sm, bw

    def _sample(self):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_idx)
        while self._running:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._sm_samples.append(util.gpu)
                self._bw_samples.append(util.memory)
            except Exception:
                pass
            time.sleep(0.05)


hw_sampler = HWSampler()

# ── Argument Parsing ───────────────────────────────────────────────────────────
DEFAULT_RATIOS = [0.2, 0.3, 0.5]

parser = argparse.ArgumentParser(description="MMMU evaluation with DivPrune.")
parser.add_argument("--subjects", nargs="+", default=None)
parser.add_argument("--split", default=SPLIT, choices=["validation", "dev", "test"])
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--max_samples", type=int, default=1000,
                    help="Max total samples to evaluate across all subjects (default: 1000).")
parser.add_argument("--output", default="results/divprune/mmmu_results.json")
parser.add_argument("--metrics_output", default="results/divprune/inference_metrics.json")
parser.add_argument("--no_divprune", action="store_true", help="Disable DivPrune (baseline run).")
parser.add_argument("--subset_ratio", type=float, default=None,
                    help="Single fraction of image tokens to KEEP. "
                         "If omitted with --no_divprune unset, sweeps over --sweep_ratios.")
parser.add_argument("--sweep_ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                    help=f"Ratios to sweep when --subset_ratio is not set "
                         f"(default: {DEFAULT_RATIOS}).")
parser.add_argument("--verbose", action="store_true", help="Print per-sample DivPrune info.")
args = parser.parse_args()

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
subjects = args.subjects or ALL_SUBJECTS

# ── Load Model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# Apply DivPrune if not baseline and a single ratio is given
_active_ratio = None
if not args.no_divprune:
    if args.subset_ratio is not None:
        _active_ratio = args.subset_ratio
    elif len(args.sweep_ratios) == 1:
        _active_ratio = args.sweep_ratios[0]

if _active_ratio is not None:
    divprune_cfg = DivPruneConfig(use_divprune=True, divprune_r=_active_ratio, verbose=args.verbose)
    model = apply_divprune_to_qwen(model, divprune_cfg)
    print(f"[DivPrune] Applied: ratio={_active_ratio}")
else:
    divprune_cfg = None
    if not args.no_divprune:
        print(f"[DivPrune] Sweep mode: ratios will be applied per-run: {args.sweep_ratios}")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)
print(f"Model ready on {next(model.parameters()).device}\n")

total_params = sum(p.numel() for p in model.parameters())
print(f"[metrics] Total parameters: {total_params/1e9:.2f}B")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# ── Vision-encoder hook ────────────────────────────────────────────────────────
_QWEN_VL_IMAGE_TOKEN_ID = 151655
_vision_start_time : float = 0.0
_vision_end_time   : float = 0.0

def _venc_pre_hook(module, input):
    global _vision_start_time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _vision_start_time = time.perf_counter()

def _venc_post_hook(module, input, output):
    global _vision_end_time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _vision_end_time = time.perf_counter()

_vision_hooks = []
if hasattr(model.model, "visual"):
    _vision_hooks.append(model.model.visual.register_forward_pre_hook(_venc_pre_hook))
    _vision_hooks.append(model.model.visual.register_forward_hook(_venc_post_hook))
    print("[metrics] Vision encoder hooks registered on model.model.visual")
elif hasattr(model, "visual"):
    _vision_hooks.append(model.visual.register_forward_pre_hook(_venc_pre_hook))
    _vision_hooks.append(model.visual.register_forward_hook(_venc_post_hook))
    print("[metrics] Vision encoder hooks registered on model.visual")
else:
    print("[metrics] Warning: vision encoder not found — vision encoder timing unavailable")


def count_image_tokens(input_ids: torch.Tensor) -> int:
    count = int((input_ids == _QWEN_VL_IMAGE_TOKEN_ID).sum().item())
    if count > 0:
        return count
    try:
        tok_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if tok_id is not None and tok_id != processor.tokenizer.unk_token_id:
            count = int((input_ids == tok_id).sum().item())
            if count > 0:
                return count
    except Exception:
        pass
    return 0


# ── Prompt Builder ─────────────────────────────────────────────────────────────
def build_messages(sample: dict) -> list:
    images = []
    for i in range(1, 8):
        img = sample.get(f"image_{i}")
        if img is not None:
            images.append(img)

    question_text = sample["question"]
    options_raw   = sample["options"]
    options       = eval(options_raw) if isinstance(options_raw, str) else options_raw
    option_labels = ["A", "B", "C", "D", "E"]
    options_str   = "\n".join(f"{option_labels[i]}) {opt}" for i, opt in enumerate(options))

    content = []
    parts   = re.split(r"<image \d+>", question_text)
    img_idx = 0

    for part in parts:
        if part.strip():
            content.append({"type": "text", "text": part})
        if img_idx < len(images):
            content.append({"type": "image", "image": images[img_idx]})
            img_idx += 1
    while img_idx < len(images):
        content.append({"type": "image", "image": images[img_idx]}); img_idx += 1
    content.append({
        "type": "text",
        "text": (f"\nOptions:\n{options_str}\n\n"
                 "Answer with only the letter of the correct option (A, B, C, or D)."),
    })
    return [{"role": "user", "content": content}]


def extract_answer(text: str) -> str:
    text = text.strip()
    if text and text[0].upper() in "ABCDE":
        return text[0].upper()
    m = re.search(r"\b([A-E])\b", text.upper())
    return m.group(1) if m else text.strip().upper()[:1]


# ── Instrumented Inference ─────────────────────────────────────────────────────
def run_inference_with_metrics(messages: list) -> tuple:
    m = SampleMetrics()
    global _vision_start_time, _vision_end_time

    t0 = time.perf_counter()
    images = []
    for msg in messages:
        for c in msg["content"]:
            if c["type"] == "image":
                images.append(c["image"])

    image_inputs = processor.image_processor(images, return_tensors="pt")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    text_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {**text_inputs, **image_inputs}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    m.image_preproc_ms = (t1 - t0) * 1e3
    m.tokenize_ms      = (t2 - t1) * 1e3
    m.input_tokens     = inputs["input_ids"].shape[1]
    m.image_tokens     = count_image_tokens(inputs["input_ids"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _vision_start_time = _vision_end_time = 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()

    hw_sampler.start()

    if is_divprune_active(model):
        dp = get_divprune_stats(model)
        if dp.pruning_applied:
            m.image_tokens = dp.kept_image_tokens

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill_end = time.perf_counter()

    m.prefill_ms        = (t_prefill_end - t_prefill_start) * 1e3
    m.ttft_ms           = m.prefill_ms
    m.vision_encoder_ms = (_vision_end_time - _vision_start_time) * 1e3 if _vision_end_time > 0 else 0.0
    m.llm_ms            = m.prefill_ms - m.vision_encoder_ms

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_decode_start = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_decode_end = time.perf_counter()

    sm_util, bw_util = hw_sampler.stop()

    m.decode_ms           = (t_decode_end - t_decode_start) * 1e3
    m.e2e_ms              = m.image_preproc_ms + m.tokenize_ms + m.decode_ms
    m.output_tokens       = generated_ids.shape[1] - m.input_tokens
    m.avg_sm_util_pct     = sm_util
    m.avg_mem_bw_util_pct = bw_util

    if torch.cuda.is_available():
        m.peak_vram_bytes = torch.cuda.max_memory_allocated()

    m.kv_cache_est_bytes  = estimate_kv_cache_bytes(model.config, m.input_tokens)
    m.prefill_flops       = estimate_prefill_flops(model.config, m.input_tokens)

    if GPU_PEAK_TFLOPS and m.prefill_ms > 0:
        achieved_tflops = m.prefill_flops / (m.prefill_ms / 1e3)
        m.mfu = achieved_tflops / GPU_PEAK_TFLOPS * 100.0

    model_bytes    = total_params * DTYPE_BYTES
    bytes_accessed = model_bytes + m.kv_cache_est_bytes
    if bytes_accessed > 0:
        m.arithmetic_intensity = m.prefill_flops / bytes_accessed

    if m.prefill_ms > 0:
        m.prefill_tok_per_s = m.input_tokens / (m.prefill_ms / 1e3)
    if m.output_tokens > 0 and m.decode_ms > 0:
        m.decode_tok_per_s  = m.output_tokens / (m.decode_ms / 1e3)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0]
    return text, m


# ── Aggregation ────────────────────────────────────────────────────────────────
def _mean(lst): return sum(lst) / len(lst) if lst else 0.0
def _p95(lst):
    if not lst: return 0.0
    s = sorted(lst); idx = int(math.ceil(0.95 * len(s))) - 1
    return s[max(idx, 0)]

def aggregate_metrics(all_m: list) -> AggregatedMetrics:
    if not all_m: return AggregatedMetrics()
    a = AggregatedMetrics(n_samples=len(all_m))
    a.mean_e2e_ms          = _mean([m.e2e_ms          for m in all_m])
    a.mean_ttft_ms         = _mean([m.ttft_ms         for m in all_m])
    tpots = [m.decode_ms / m.output_tokens for m in all_m if m.output_tokens > 0]
    a.mean_tpot_ms         = _mean(tpots)
    a.mean_prefill_ms      = _mean([m.prefill_ms      for m in all_m])
    a.mean_decode_ms       = _mean([m.decode_ms       for m in all_m])
    a.mean_prefill_tok_s   = _mean([m.prefill_tok_per_s for m in all_m])
    a.mean_decode_tok_s    = _mean([m.decode_tok_per_s  for m in all_m if m.decode_tok_per_s > 0])
    a.mean_image_tokens    = _mean([m.image_tokens    for m in all_m])
    a.mean_input_tokens    = _mean([m.input_tokens    for m in all_m])
    a.mean_output_tokens   = _mean([m.output_tokens   for m in all_m])
    a.mean_peak_vram_mb    = _mean([m.peak_vram_bytes / 1e6 for m in all_m])
    a.mean_kvcache_mb      = _mean([m.kv_cache_est_bytes / 1e6 for m in all_m])
    a.mean_mfu_pct         = _mean([m.mfu             for m in all_m if m.mfu > 0])
    a.mean_arith_intensity = _mean([m.arithmetic_intensity for m in all_m if m.arithmetic_intensity > 0])
    a.mean_sm_util_pct     = _mean([m.avg_sm_util_pct for m in all_m if m.avg_sm_util_pct > 0])
    a.mean_mem_bw_util_pct = _mean([m.avg_mem_bw_util_pct for m in all_m if m.avg_mem_bw_util_pct > 0])
    a.p95_e2e_ms           = _p95([m.e2e_ms    for m in all_m])
    a.p95_ttft_ms          = _p95([m.ttft_ms   for m in all_m])
    a.p95_decode_tok_s     = _p95([m.decode_tok_per_s for m in all_m if m.decode_tok_per_s > 0])
    return a


# ── Discipline map ─────────────────────────────────────────────────────────────
discipline_map = {
    "Art & Design":                ["Art", "Art_Theory", "Design", "Music"],
    "Business":                    ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science":                     ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health & Medicine":           ["Basic_Medical_Science", "Clinical_Medicine",
                                    "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"],
    "Humanities & Social Science": ["History", "Literature", "Psychology", "Sociology"],
    "Tech & Engineering":          ["Agriculture", "Architecture_and_Engineering", "Computer_Science",
                                    "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering"],
}


# ── Single-ratio evaluation run ────────────────────────────────────────────────
def run_evaluation(ratio, output_path, metrics_output_path):
    """Run a full MMMU evaluation for a given DivPrune ratio (or None for baseline)."""
    tag = "baseline" if ratio is None else f"r{ratio:.2f}".replace(".", "p")
    print("\n" + "=" * 80)
    print(f"MMMU Evaluation — DivPrune  |  ratio={'baseline' if ratio is None else ratio}")
    print("=" * 80)

    if ratio is not None and not is_divprune_active(model):
        cfg = DivPruneConfig(use_divprune=True, divprune_r=ratio, verbose=args.verbose)
        apply_divprune_to_qwen(model, cfg)

    all_results        = {}
    subject_acc        = {}
    all_sample_metrics = []
    total_correct      = 0
    total_count        = 0
    global_sample_idx  = 0
    samples_remaining  = args.max_samples

    for subject in subjects:
        if samples_remaining <= 0:
            break

        print(f"\n{'='*60}\nSubject: {subject}\n{'='*60}")
        try:
            dataset = load_dataset(DATASET_ID, subject, split=args.split,
                                   trust_remote_code=True, verification_mode="no_checks")
        except Exception as e:
            print(f"  Could not load {subject}: {e}"); continue

        records = []
        correct = 0

        for sample in tqdm(dataset, desc=subject, ncols=80):
            if samples_remaining <= 0:
                break
            if sample.get("question_type") == "open":
                continue

            gt_answer = sample["answer"].strip().upper()
            try:
                messages         = build_messages(sample)
                raw_pred, metrics = run_inference_with_metrics(messages)
                pred             = extract_answer(raw_pred)
            except Exception as e:
                print(f"  Error on {sample['id']}: {e}")
                pred    = ""
                metrics = SampleMetrics()

            is_corr = pred == gt_answer
            if is_corr: correct += 1

            divprune_info = {}
            if is_divprune_active(model):
                stats = get_divprune_stats(model)
                divprune_info = {
                    "pruning_applied":    stats.pruning_applied,
                    "total_image_tokens": stats.total_image_tokens,
                    "kept_image_tokens":  stats.kept_image_tokens,
                    "original_seq_len":   stats.original_seq_len,
                    "pruned_seq_len":     stats.pruned_seq_len,
                }

            records.append({
                "id"           : sample["id"],
                "gt"           : gt_answer,
                "pred"         : pred,
                "correct"      : is_corr,
                "question_type": sample.get("question_type", ""),
                "difficulty"   : sample.get("topic_difficulty", ""),
                "metrics"      : asdict(metrics),
                "divprune"     : divprune_info,
            })
            all_sample_metrics.append(metrics)
            global_sample_idx += 1
            samples_remaining -= 1

        n   = len(records)
        acc = correct / n if n > 0 else 0.0
        subject_acc[subject] = acc
        all_results[subject] = records
        total_correct       += correct
        total_count         += n
        print(f"  Accuracy: {correct}/{n} = {acc:.2%}")

        subj_metrics = all_sample_metrics[-n:]
        if subj_metrics:
            ag = aggregate_metrics(subj_metrics)
            print(f"  [perf] TTFT={ag.mean_ttft_ms:.0f}ms  TPOT={ag.mean_tpot_ms:.1f}ms/tok  "
                  f"decode={ag.mean_decode_tok_s:.1f}tok/s  "
                  f"VRAM={ag.mean_peak_vram_mb:.0f}MB  "
                  f"MFU={ag.mean_mfu_pct:.1f}%")

    overall_acc = total_correct / total_count if total_count else 0.0
    global_perf = aggregate_metrics(all_sample_metrics)

    discipline_acc = {}
    for discipline, subs in discipline_map.items():
        d_correct = sum(sum(1 for r in all_results.get(s, []) if r["correct"]) for s in subs)
        d_total   = sum(len(all_results.get(s, [])) for s in subs)
        discipline_acc[discipline] = d_correct / d_total if d_total else 0.0

    # Save accuracy results
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "model"        : MODEL_ID,
        "split"        : args.split,
        "divprune"     : ratio is not None,
        "subset_ratio" : ratio,
        "max_samples"  : args.max_samples,
        "overall"      : {"correct": total_correct, "total": total_count, "accuracy": overall_acc},
        "by_discipline": {k: f"{v:.2%}" for k, v in discipline_acc.items()},
        "by_subject"   : {k: f"{v:.2%}" for k, v in subject_acc.items()},
        "per_sample"   : all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Save metrics results
    metrics_path = Path(metrics_output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_data = {
        "model"           : MODEL_ID,
        "gpu"             : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_peak_tflops" : GPU_PEAK_TFLOPS,
        "gpu_peak_bw_gbps": GPU_PEAK_BW_GBPS,
        "total_params_B"  : round(total_params / 1e9, 3),
        "aggregated"      : asdict(global_perf),
        "divprune_config" : asdict(get_divprune_config(model)) if is_divprune_active(model) else None,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2, default=str)

    g = global_perf
    print(f"""
{'='*60}
MMMU EVALUATION SUMMARY  [{tag}]
{'='*60}
Overall Accuracy  : {total_correct}/{total_count} = {overall_acc:.2%}
DivPrune          : {'enabled' if ratio is not None else 'disabled'} (ratio={ratio})

By Discipline:""")
    for discipline, acc in discipline_acc.items():
        print(f"  {discipline:<35} {acc:.2%}")

    print(f"""
{'='*60}
INFERENCE PERFORMANCE SUMMARY  (n={g.n_samples} samples)
{'='*60}

── Latency ──────────────────────────────────────────────
  TTFT  (mean / p95)       : {g.mean_ttft_ms:.1f} ms  /  {g.p95_ttft_ms:.1f} ms
  TPOT  (mean)             : {g.mean_tpot_ms:.2f} ms/tok
  Prefill (mean)           : {g.mean_prefill_ms:.1f} ms
  Decode  (mean)           : {g.mean_decode_ms:.1f} ms
  End-to-End (mean / p95)  : {g.mean_e2e_ms:.1f} ms  /  {g.p95_e2e_ms:.1f} ms

── Throughput ───────────────────────────────────────────
  Prefill                  : {g.mean_prefill_tok_s:.0f} tok/s
  Decode  (mean / p95)     : {g.mean_decode_tok_s:.1f}  /  {g.p95_decode_tok_s:.1f} tok/s

── Token Counts ─────────────────────────────────────────
  Input tokens (mean)      : {g.mean_input_tokens:.0f}
  Image tokens (mean)      : {g.mean_image_tokens:.0f}
  Output tokens (mean)     : {g.mean_output_tokens:.1f}

── Memory ───────────────────────────────────────────────
  Peak VRAM (mean)         : {g.mean_peak_vram_mb:.0f} MB
  KV-cache est (mean)      : {g.mean_kvcache_mb:.1f} MB

── Compute ──────────────────────────────────────────────
  MFU (mean)               : {g.mean_mfu_pct:.2f}%
  Arithmetic Intensity     : {g.mean_arith_intensity:.1f} FLOPs/byte

── Hardware Utilisation ─────────────────────────────────
  SM Utilisation (mean)    : {g.mean_sm_util_pct:.1f}%
  Mem-BW Utilisation (mean): {g.mean_mem_bw_util_pct:.1f}%

Full accuracy  → {out_path}
Full metrics   → {metrics_path}
""")

    if ratio is not None and is_divprune_active(model):
        remove_divprune(model)

    return overall_acc


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if args.no_divprune:
        ratios = [None]
    elif args.subset_ratio is not None:
        ratios = [args.subset_ratio]
    else:
        ratios = args.sweep_ratios

    print(f"\nRatios to evaluate: {['baseline' if r is None else r for r in ratios]}\n")

    summary = []
    for ratio in ratios:
        tag = "baseline" if ratio is None else f"r{ratio:.2f}".replace(".", "p")
        out  = args.output.replace(".json", f"_{tag}.json") if len(ratios) > 1 else args.output
        mout = args.metrics_output.replace(".json", f"_{tag}.json") if len(ratios) > 1 else args.metrics_output
        acc  = run_evaluation(ratio, out, mout)
        summary.append({"ratio": ratio, "accuracy": acc})

    if len(ratios) > 1:
        print("\n" + "=" * 80)
        print("SWEEP SUMMARY")
        print("=" * 80)
        print(f"{'Ratio':<10}  {'Accuracy':>10}")
        print("-" * 22)
        for entry in summary:
            label = "baseline" if entry["ratio"] is None else f"{entry['ratio']:.2f}"
            print(f"{label:<10}  {entry['accuracy']:>10.1%}")
        print("=" * 80)


if __name__ == "__main__":
    main()

# ── Cleanup hooks ──────────────────────────────────────────────────────────────
for h in _vision_hooks:
    h.remove()
