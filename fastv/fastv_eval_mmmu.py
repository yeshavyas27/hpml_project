"""
evaluate_mmmu.py  — with comprehensive inference metrics
Evaluates Qwen3-VL-4B-Instruct on the MMMU validation set (900 questions).
Saves per-subject results + overall accuracy + perf metrics to results/mmmu_results.json

Metrics collected
─────────────────
Compute        : FLOPs/MACs (calflops), MFU, arithmetic intensity (prefill)
Memory         : peak VRAM, KV-cache estimate, memory-BW utilisation
Latency        : TTFT, TPOT, prefill_ms, decode_ms, e2e_ms, image-preproc_ms
Throughput     : prefill tok/s, decode tok/s
Vision-specific: image token count, vision-encoder ms, LLM ms
Hardware       : SM utilisation, memory-BW, occupancy  (via pynvml + torch.profiler)
"""

import os, gc, json, re, time, argparse, math, threading
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

# ── Import the correct class name for Qwen3-VL ──────────────────────────────
# Requires transformers >= 4.57.0
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from fastv_qwen import (
    FastVConfig, apply_fastv_to_qwen, remove_fastv,
    get_fastv_stats, is_fastv_active, get_fastv_config)

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    from calflops import calculate_flops
    HAS_CALFLOPS = True
except ImportError:
    HAS_CALFLOPS = False
    print("[metrics] calflops not found – FLOPs via manual estimation")

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
RESULTS_DIR = Path("results")

ALL_SUBJECTS = [
    "Accounting","Agriculture","Architecture_and_Engineering","Art","Art_Theory",
    "Basic_Medical_Science","Biology","Chemistry","Clinical_Medicine","Computer_Science",
    "Design","Diagnostics_and_Laboratory_Medicine","Economics","Electronics",
    "Energy_and_Power","Finance","Geography","History","Literature","Manage",
    "Marketing","Materials","Math","Mechanical_Engineering","Music","Pharmacy",
    "Physics","Psychology","Public_Health","Sociology",
]

# ── GPU constants ──────────────────────────────────────────────────────────────
GPU_PEAK_TFLOPS   = None   # filled at runtime
GPU_PEAK_BW_GBPS  = None   # GB/s memory bandwidth
DTYPE_BYTES       = 2      # bf16

def _detect_gpu_specs():
    global GPU_PEAK_TFLOPS, GPU_PEAK_BW_GBPS
    if not torch.cuda.is_available():
        return
    name = torch.cuda.get_device_name(0).upper()
    # rough lookup – extend as needed
    specs = {
        "A100": (312, 2000), "H100": (989, 3350),
        "A10":  (125, 600),  "A10G": (125, 600),
        "3090": (35.6, 936), "4090": (82.6, 1008),
        "3080": (29.8, 760), "4080": (49.0, 716),
        "T4":   (65, 300),   "V100": (125, 900),
    }
    for key, (tflops, bw) in specs.items():
        if key in name:
            GPU_PEAK_TFLOPS  = tflops * 1e12   # ops/s
            GPU_PEAK_BW_GBPS = bw * 1e9        # bytes/s
            print(f"[metrics] Detected GPU: {name}  |  peak {tflops} TFLOPS  |  {bw} GB/s BW")
            return
    # fallback – query via pynvml
    if HAS_PYNVML:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        GPU_PEAK_BW_GBPS = 900e9  # conservative default
        GPU_PEAK_TFLOPS  = 100e12
        print(f"[metrics] Unknown GPU; using conservative defaults")

_detect_gpu_specs()

# ── Dataclasses ────────────────────────────────────────────────────────────────
@dataclass
class SampleMetrics:
    # Latency (ms)
    image_preproc_ms   : float = 0.0
    tokenize_ms        : float = 0.0
    ttft_ms            : float = 0.0   # time-to-first-token  = prefill
    prefill_ms         : float = 0.0
    decode_ms          : float = 0.0
    e2e_ms             : float = 0.0
    vision_encoder_ms  : float = 0.0
    llm_ms             : float = 0.0

    # Token counts
    input_tokens       : int   = 0
    image_tokens       : int   = 0
    output_tokens      : int   = 0

    # Memory (bytes)
    peak_vram_bytes    : int   = 0
    kv_cache_est_bytes : int   = 0

    # Compute
    prefill_flops      : float = 0.0   # estimated
    mfu                : float = 0.0
    arithmetic_intensity: float = 0.0  # FLOPs / byte

    # Throughput
    prefill_tok_per_s  : float = 0.0
    decode_tok_per_s   : float = 0.0

    # HW (sampled during decode)
    avg_sm_util_pct    : float = 0.0
    avg_mem_bw_util_pct: float = 0.0


@dataclass
class AggregatedMetrics:
    n_samples: int = 0
    # means
    mean_e2e_ms          : float = 0.0
    mean_ttft_ms         : float = 0.0
    mean_tpot_ms         : float = 0.0   # ms per output token
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
    # p95
    p95_e2e_ms           : float = 0.0
    p95_ttft_ms          : float = 0.0
    p95_decode_tok_s     : float = 0.0


# ── Helper: estimate KV-cache size ────────────────────────────────────────────
def estimate_kv_cache_bytes(model_cfg, seq_len: int) -> int:
    """2 * layers * heads * head_dim * seq_len * bytes"""
    try:
        # For Qwen3-VL, the text config is nested under model_cfg.text_config
        text_cfg = getattr(model_cfg, 'text_config', model_cfg)
        n_layers = text_cfg.num_hidden_layers
        n_kv_heads = getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)
        head_dim = getattr(text_cfg, 'head_dim', text_cfg.hidden_size // text_cfg.num_attention_heads)
        return 2 * n_layers * n_kv_heads * head_dim * seq_len * DTYPE_BYTES
    except Exception:
        return 0


# ── Helper: estimate prefill FLOPs ────────────────────────────────────────────
def estimate_prefill_flops(model_cfg, seq_len: int) -> float:
    text_cfg = getattr(model_cfg, "text_config", model_cfg)
    layers = text_cfg.num_hidden_layers
    hidden_dim = text_cfg.hidden_size

    attn_flops = 2 * layers * (seq_len ** 2) * hidden_dim
    mlp_flops  = 4 * layers * seq_len * (hidden_dim ** 2)

    return attn_flops + mlp_flops

# ── Helper: hardware utilisation sampler ──────────────────────────────────────
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
        """Returns (mean_sm_util, mean_mem_bw_util) as percentages."""
        if not HAS_PYNVML:
            return 0.0, 0.0
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        sm  = sum(self._sm_samples) / len(self._sm_samples)  if self._sm_samples else 0.0
        bw  = sum(self._bw_samples) / len(self._bw_samples)  if self._bw_samples else 0.0
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
parser = argparse.ArgumentParser()
parser.add_argument("--subjects", nargs="+", default=None)
parser.add_argument("--split", default=SPLIT, choices=["validation","dev","test"])
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--output", default="results/mmmu_results.json")
parser.add_argument("--metrics_output", default="results/inference_metrics.json")
parser.add_argument("--profile_every", type=int, default=50,
                    help="Run torch.profiler every N samples (0=disable)")
parser.add_argument("--no_fastv", action="store_true", help="Disable FastV pruning")
parser.add_argument("--fastv_k", type=int, default=3, help="FastV: prune after layer K")
parser.add_argument("--fastv_r", type=float, default=0.5, help="FastV: ratio of image tokens to keep")
args = parser.parse_args()

RESULTS_DIR.mkdir(exist_ok=True)
subjects = args.subjects or ALL_SUBJECTS

# ── Load Model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # NOTE: if you want attention weights for FastV importance scoring
    # (slower but potentially more accurate), uncomment:
    # attn_implementation="eager",
)
model.eval()

FASTV_CONFIG = FastVConfig(
      use_fastv = True,
      fastv_k   = 3,      # layer after which to prune; try 1,2,3,5
      fastv_r   = 0.5,    # fraction of image tokens to keep
      verbose   = True,   # KEEP TRUE until you see "[FastV] ✓" in output
  )
if FASTV_CONFIG.use_fastv:
    model = apply_fastv_to_qwen(model, FASTV_CONFIG)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)
print(f"Model ready on {next(model.parameters()).device}\n")

# ── Model parameter count ──────────────────────────────────────────────────────
total_params = sum(p.numel() for p in model.parameters())
print(f"[metrics] Total parameters: {total_params/1e9:.2f}B")

# Reset peak memory tracker
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# ── Vision-encoder hook ────────────────────────────────────────────────────────
# Qwen3-VL image token ID (the <|image_pad|> / vision placeholder token).
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

# ✅ FIXED: Register hooks on model.model.visual (NOT model.model.visual_encoder)
# The Qwen3-VL architecture uses model.model.visual for the vision encoder.
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
    """Count image tokens by scanning input_ids for the Qwen-VL image pad token."""
    count = int((input_ids == _QWEN_VL_IMAGE_TOKEN_ID).sum().item())
    if count > 0:
        return count

    # Fallback: look up the token ID from the processor vocab at runtime
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
    option_labels = ["A","B","C","D","E"]
    options_str   = "\n".join(f"{option_labels[i]}) {opt}" for i, opt in enumerate(options))

    content  = []
    parts    = re.split(r"<image \d+>", question_text)
    img_idx  = 0

    for part in parts:
        if part.strip():
            content.append({"type":"text","text":part})
        if img_idx < len(images):
            content.append({"type":"image","image":images[img_idx]})
            img_idx += 1
    while img_idx < len(images):
        content.append({"type":"image","image":images[img_idx]}); img_idx += 1
    content.append({
        "type":"text",
        "text":(f"\nOptions:\n{options_str}\n\n"
                "Answer with only the letter of the correct option (A, B, C, or D)."),
    })
    return [{"role":"user","content":content}]


def extract_answer(text: str) -> str:
    text = text.strip()
    if text and text[0].upper() in "ABCDE":
        return text[0].upper()
    m = re.search(r"\b([A-E])\b", text.upper())
    return m.group(1) if m else text.strip().upper()[:1]


# ── Instrumented Inference ─────────────────────────────────────────────────────
def run_inference_with_metrics(messages: list, sample_idx: int = 0) -> tuple:
    """
    Returns (decoded_text, SampleMetrics).
    """
    m = SampleMetrics()
    global _vision_start_time, _vision_end_time

    # ── 1. Image preprocessing ─────────────────────────────────────────────
    t0 = time.perf_counter()

    # Extract images
    images = []
    for msg in messages:
        for c in msg["content"]:
            if c["type"] == "image":
                images.append(c["image"])

    # Run image preprocessing explicitly
    image_inputs = processor.image_processor(images, return_tensors="pt")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Now tokenize text separately
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

    # ── Count image tokens directly from input_ids ────────────────────────
    m.image_tokens = count_image_tokens(inputs["input_ids"])

    # ── 3. Reset memory peak before forward ───────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── 4. Prefill (first token) ───────────────────────────────────────────
    _vision_start_time = _vision_end_time = 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()

    hw_sampler.start()

    if is_fastv_active(model):
        fv = get_fastv_stats(model)
        if fv.pruning_applied:
            m.image_tokens = fv.kept_image_tokens  # override: kept, not total

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill_end = time.perf_counter()

    m.prefill_ms        = (t_prefill_end - t_prefill_start) * 1e3
    m.ttft_ms           = m.prefill_ms
    m.vision_encoder_ms = (_vision_end_time - _vision_start_time) * 1e3 if _vision_end_time > 0 else 0.0
    m.llm_ms            = m.prefill_ms - m.vision_encoder_ms

    # ── 5. Full decode ─────────────────────────────────────────────────────
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

    m.decode_ms          = (t_decode_end - t_decode_start) * 1e3
    m.e2e_ms             = m.image_preproc_ms + m.tokenize_ms + m.decode_ms
    m.output_tokens      = generated_ids.shape[1] - m.input_tokens
    m.avg_sm_util_pct    = sm_util
    m.avg_mem_bw_util_pct= bw_util

    # ── 6. VRAM ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        m.peak_vram_bytes = torch.cuda.max_memory_allocated()

    # ── 7. KV-cache estimate ───────────────────────────────────────────────
    m.kv_cache_est_bytes = estimate_kv_cache_bytes(model.config, m.input_tokens)

    # ── 8. FLOPs / MFU / Arithmetic Intensity ─────────────────────────────
    m.prefill_flops = estimate_prefill_flops(model.config, m.input_tokens)

    if GPU_PEAK_TFLOPS and m.prefill_ms > 0:
        achieved_tflops = m.prefill_flops / (m.prefill_ms / 1e3)
        m.mfu = achieved_tflops / GPU_PEAK_TFLOPS * 100.0  # %

    # Arithmetic intensity: FLOPs / bytes_accessed
    model_bytes = total_params * DTYPE_BYTES
    bytes_accessed = model_bytes + m.kv_cache_est_bytes
    if bytes_accessed > 0:
        m.arithmetic_intensity = m.prefill_flops / bytes_accessed

    # ── 9. Throughput ──────────────────────────────────────────────────────
    if m.prefill_ms > 0:
        m.prefill_tok_per_s = m.input_tokens / (m.prefill_ms / 1e3)
    if m.output_tokens > 0 and m.decode_ms > 0:
        m.decode_tok_per_s  = m.output_tokens / (m.decode_ms / 1e3)

    # ── 10. Decode response ────────────────────────────────────────────────
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)[0]
    return text, m


# ── Aggregation ────────────────────────────────────────────────────────────────
def _mean(lst): return sum(lst)/len(lst) if lst else 0.0
def _p95(lst):
    if not lst: return 0.0
    s = sorted(lst); idx = int(math.ceil(0.95*len(s))) - 1
    return s[max(idx,0)]

def aggregate_metrics(all_m: list) -> AggregatedMetrics:
    if not all_m: return AggregatedMetrics()
    a = AggregatedMetrics(n_samples=len(all_m))
    a.mean_e2e_ms          = _mean([m.e2e_ms          for m in all_m])
    a.mean_ttft_ms         = _mean([m.ttft_ms         for m in all_m])
    tpots = [m.decode_ms/m.output_tokens for m in all_m if m.output_tokens > 0]
    a.mean_tpot_ms         = _mean(tpots)
    a.mean_prefill_ms      = _mean([m.prefill_ms      for m in all_m])
    a.mean_decode_ms       = _mean([m.decode_ms       for m in all_m])
    a.mean_prefill_tok_s   = _mean([m.prefill_tok_per_s for m in all_m])
    a.mean_decode_tok_s    = _mean([m.decode_tok_per_s  for m in all_m if m.decode_tok_per_s>0])
    a.mean_image_tokens    = _mean([m.image_tokens    for m in all_m])
    a.mean_input_tokens    = _mean([m.input_tokens    for m in all_m])
    a.mean_output_tokens   = _mean([m.output_tokens   for m in all_m])
    a.mean_peak_vram_mb    = _mean([m.peak_vram_bytes /1e6 for m in all_m])
    a.mean_kvcache_mb      = _mean([m.kv_cache_est_bytes/1e6 for m in all_m])
    a.mean_mfu_pct         = _mean([m.mfu             for m in all_m if m.mfu>0])
    a.mean_arith_intensity = _mean([m.arithmetic_intensity for m in all_m if m.arithmetic_intensity>0])
    a.mean_sm_util_pct     = _mean([m.avg_sm_util_pct for m in all_m if m.avg_sm_util_pct>0])
    a.mean_mem_bw_util_pct = _mean([m.avg_mem_bw_util_pct for m in all_m if m.avg_mem_bw_util_pct>0])
    a.p95_e2e_ms           = _p95([m.e2e_ms    for m in all_m])
    a.p95_ttft_ms          = _p95([m.ttft_ms   for m in all_m])
    a.p95_decode_tok_s     = _p95([m.decode_tok_per_s for m in all_m if m.decode_tok_per_s>0])
    return a


# ── Discipline map ─────────────────────────────────────────────────────────────
discipline_map = {
    "Art & Design":               ["Art","Art_Theory","Design","Music"],
    "Business":                   ["Accounting","Economics","Finance","Manage","Marketing"],
    "Science":                    ["Biology","Chemistry","Geography","Math","Physics"],
    "Health & Medicine":          ["Basic_Medical_Science","Clinical_Medicine",
                                   "Diagnostics_and_Laboratory_Medicine","Pharmacy","Public_Health"],
    "Humanities & Social Science":["History","Literature","Psychology","Sociology"],
    "Tech & Engineering":         ["Agriculture","Architecture_and_Engineering","Computer_Science",
                                   "Electronics","Energy_and_Power","Materials","Mechanical_Engineering"],
}

# ── Main Loop ──────────────────────────────────────────────────────────────────
all_results       = {}
subject_acc       = {}
all_sample_metrics: list = []
total_correct     = 0
total_count       = 0
global_sample_idx = 0

for subject in subjects:
    print(f"\n{'='*60}\nSubject: {subject}\n{'='*60}")
    try:
        dataset = load_dataset(DATASET_ID, subject, split=args.split, trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠ Could not load {subject}: {e}"); continue

    records = []
    correct = 0

    for sample in tqdm(dataset, desc=subject, ncols=80):
        if sample.get("question_type") == "open":
            continue
        gt_answer = sample["answer"].strip().upper()
        try:
            messages          = build_messages(sample)
            raw_pred, metrics = run_inference_with_metrics(messages, global_sample_idx)
            pred              = extract_answer(raw_pred)
        except Exception as e:
            print(f"  Error on {sample['id']}: {e}")
            pred    = ""
            metrics = SampleMetrics()

        is_correct = pred == gt_answer
        if is_correct: correct += 1

        records.append({
            "id"           : sample["id"],
            "gt"           : gt_answer,
            "pred"         : pred,
            "correct"      : is_correct,
            "question_type": sample.get("question_type",""),
            "difficulty"   : sample.get("topic_difficulty",""),
            "metrics"      : asdict(metrics),
            "fastv": asdict(get_fastv_stats(model)) if is_fastv_active(model) else {}
        })
        all_sample_metrics.append(metrics)
        global_sample_idx += 1

    n   = len(records)
    acc = correct / n if n > 0 else 0.0
    subject_acc[subject]  = acc
    all_results[subject]  = records
    total_correct        += correct
    total_count          += n
    print(f"  Accuracy: {correct}/{n} = {acc:.2%}")

    # Print running per-subject metric summary
    subj_metrics = all_sample_metrics[-n:]
    if subj_metrics:
        ag = aggregate_metrics(subj_metrics)
        print(f"  [perf] TTFT={ag.mean_ttft_ms:.0f}ms  TPOT={ag.mean_tpot_ms:.1f}ms/tok  "
              f"decode={ag.mean_decode_tok_s:.1f}tok/s  "
              f"VRAM={ag.mean_peak_vram_mb:.0f}MB  "
              f"MFU={ag.mean_mfu_pct:.1f}%")

# ── Global aggregation ─────────────────────────────────────────────────────────
overall_acc    = total_correct / total_count if total_count else 0.0
global_perf    = aggregate_metrics(all_sample_metrics)

# ── Discipline accuracy ────────────────────────────────────────────────────────
discipline_acc = {}
for discipline, subs in discipline_map.items():
    d_correct = sum(sum(1 for r in all_results.get(s,[]) if r["correct"]) for s in subs)
    d_total   = sum(len(all_results.get(s,[])) for s in subs)
    discipline_acc[discipline] = d_correct / d_total if d_total else 0.0

# ── Save accuracy results ──────────────────────────────────────────────────────
output = {
    "model"       : MODEL_ID,
    "split"       : args.split,
    "fastv"       : not args.no_fastv,
    "fastv_k"     : args.fastv_k,
    "fastv_r"     : args.fastv_r,
    "overall"     : {"correct":total_correct,"total":total_count,"accuracy":overall_acc},
    "by_discipline": {k:f"{v:.2%}" for k,v in discipline_acc.items()},
    "by_subject"  : {k:f"{v:.2%}" for k,v in subject_acc.items()},
    "per_sample"  : all_results,
}
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path,"w") as f:
    json.dump(output, f, indent=2)

# ── Save metrics results ───────────────────────────────────────────────────────
metrics_output = {
    "model"           : MODEL_ID,
    "gpu"             : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "gpu_peak_tflops" : GPU_PEAK_TFLOPS,
    "gpu_peak_bw_gbps": GPU_PEAK_BW_GBPS,
    "total_params_B"  : round(total_params/1e9, 3),
    "aggregated"      : asdict(global_perf),
    "fastv_config": asdict(get_fastv_config(model)) if is_fastv_active(model) else None,
}
metrics_path = Path(args.metrics_output)
metrics_path.parent.mkdir(parents=True, exist_ok=True)
with open(metrics_path,"w") as f:
    json.dump(metrics_output, f, indent=2, default=str)

# ── Final Summary ──────────────────────────────────────────────────────────────
g = global_perf
print(f"""
{'='*60}
MMMU EVALUATION SUMMARY
{'='*60}
Overall Accuracy  : {total_correct}/{total_count} = {overall_acc:.2%}
FastV             : {'enabled' if not args.no_fastv else 'disabled'} (k={args.fastv_k}, r={args.fastv_r})

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

# ── Cleanup hooks ──────────────────────────────────────────────────────────────
for h in _vision_hooks:
    h.remove()