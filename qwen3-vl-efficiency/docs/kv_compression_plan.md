# KV Cache Compression for Qwen3-VL — Step-by-Step Plan

This is your runbook for the "KV cache compression" part of the project.
It mirrors your project outline (Sections IV.C and the Milestone 6 items in
the task list) and maps each item to code in this repo.

---

## Goal recap

1. Reduce the GPU memory held by Qwen3-VL's KV cache during inference.
2. Do it **training-free** — no fine-tuning, no weight changes.
3. Keep accuracy close to baseline on MMMU / MathVista / MMBench /
   RealWorldQA / DocVQA.
4. Show that **modality-aware** eviction (separate image / text budgets)
   beats any uniform policy.

---

## Methods in this repo

| Method       | Policy                                    | Scorer                 | Wrapper file                     |
|--------------|-------------------------------------------|------------------------|----------------------------------|
| `h2o`        | Heavy-hitter, uniform                     | ExpectedAttention      | `src/kv_compression/h2o.py`      |
| `streaming`  | Attention sinks + recent window           | Positional only        | `src/kv_compression/streaming.py`|
| `snapkv`     | Prompt-aware (last-window attention)      | SnapKV                 | `src/kv_compression/snapkv.py`   |
| `pyramid`    | SnapKV + per-layer pyramid budget         | SnapKV                 | `src/kv_compression/snapkv.py`   |
| `modality`   | Separate image / text budgets (NEW)       | H2O or SnapKV (inner)  | `src/kv_compression/modality.py` |

All live under one dispatcher:

```python
from src.kv_compression import make_press

press = make_press("h2o",      compression_ratio=0.5)
press = make_press("streaming",compression_ratio=0.5, n_sink=4)
press = make_press("snapkv",   compression_ratio=0.5, window_size=64)
press = make_press("pyramid",  compression_ratio=0.5, beta=20)
press = make_press("modality", image_compression_ratio=0.7,
                               text_compression_ratio=0.2,
                               inner="h2o")

with press(model):
    out = model.generate(**inputs, max_new_tokens=64)
```

---

## Phase 0 — Environment (already done for you)

1. Python 3.10+, CUDA-capable GPU (A100 32 GB on NYU HPC per your setup).
2. `pip install -r requirements.txt` — `kvpress` is already listed.
3. Confirm Qwen3-VL loads:

   ```bash
   python -c "from src.load_model import load_model_and_processor; m, p = load_model_and_processor(); print(m.config.model_type)"
   ```

---

## Phase 1 — Establish the baseline (already done)

You already have baseline accuracy + latency + memory numbers in the README
table. Use those as your "no compression" row. Also run `--compression_ratio 0.0`
with each method as a sanity check — results should match the baseline within
noise.

---

## Phase 2 — Run uniform methods (H2O / StreamingLLM / SnapKV / PyramidKV)

Smoke test each method on one dataset:

```bash
for m in h2o streaming snapkv pyramid; do
    python -m eval.eval_kv_methods --method $m \
        --dataset realworldqa --compression_ratio 0.5 --max_samples 5
done
```

Full sweep (uniform methods × ratios × datasets):

```bash
MAX_SAMPLES=50 bash scripts/run_kv_sweep.sh
```

This populates `results/kv_compression/` with one JSONL per cell. Aggregate
later into a single CSV for plotting.

What to look for:

- **H2O** is the strong baseline. If something doesn't beat H2O, report it as
  a negative result.
- **StreamingLLM** should be noticeably worse on image-heavy tasks (DocVQA,
  MathVista) because it blindly evicts the middle, which is where image
  tokens live. This is expected and motivates modality-awareness.
- **SnapKV** should be especially good on VQA-style questions where the
  question explicitly points at a region of the image.
- **PyramidKV** may help or hurt — the interesting signal is whether VLM
  layers respond the same way to pyramid budgeting as text-only LLMs.

---

## Phase 3 — Run the modality-aware method (your novel contribution)

The key hypothesis: **image tokens are more evictable than text tokens, so
giving each modality its own budget should Pareto-dominate any uniform
method.**

A good grid to start with — keeps total eviction around 50% but tilts toward
image tokens:

| image_ratio | text_ratio | effective_total |
|-------------|------------|-----------------|
| 0.5         | 0.2        | ~0.48 on DocVQA |
| 0.7         | 0.2        | ~0.65 on DocVQA |
| 0.7         | 0.3        | ~0.68 on DocVQA |
| 0.8         | 0.3        | ~0.78 on DocVQA |

Run them:

```bash
for img_r in 0.3 0.5 0.7; do
  for txt_r in 0.1 0.2 0.3; do
    python -m eval.eval_kv_methods --method modality \
        --dataset docvqa \
        --image_compression_ratio $img_r \
        --text_compression_ratio  $txt_r \
        --inner h2o \
        --max_samples 50
  done
done
```

`effective_compression_ratio` is logged for every sample so you can plot
**accuracy vs effective compression** for both the uniform and modality-aware
methods on the same axes.

---

## Phase 4 — Analysis

1. **Pareto plot**: accuracy (y) vs effective compression (x), one curve per
   method, per dataset. Modality-aware should sit above the uniform curves
   on image-heavy datasets.
2. **Memory & latency plot**: peak GPU memory (GB) vs compression ratio.
   All methods should reduce memory linearly in ratio; modality-aware ties
   this saving to the image-fraction of the prompt.
3. **Prefill vs decode breakdown**: prefill latency is largely unchanged
   (compression happens at the *end* of prefill); decode latency should
   drop because the per-step attention now reads a smaller cache.
4. **Failure analysis**: pull a handful of samples where H2O got it right
   and modality-aware did not (and vice versa). Grep the predictions and
   look at the mask summaries. This goes straight into the Failure Analysis
   section of the paper.
5. **Task-wise analysis**: compression tends to hurt mathy reasoning
   (MathVista, MMMU-Math) more than visual recognition (RealWorldQA).
   Report this explicitly — it's a finding, not a bug.

---

## Phase 5 — Integration with the rest of the project

- **Visual-token reduction** (Aadarsh / Yesha): feeds *fewer* visual tokens
  into the prompt. Your KV cache compression runs on top — the remaining
  visual tokens still get an eviction policy. Run modality-aware KV
  compression on outputs of their DuetVLM / visual pruning step to measure
  the stacked effect (item #8 in the task list, "combine everything").
- **Quantization** (Adarsh, Modality-Aware InnerQ): orthogonal to eviction.
  Evict first, quantize what remains. Your ratios stay meaningful.
- **Attention analysis** (item #10): the importance scores inside
  `ExpectedAttentionPress` and the modality mask together give you
  ready-made numbers for the attention-pattern plots.

---

## File map

```
src/
  kv_compression/
    __init__.py        # make_press dispatcher + factories
    h2o.py             # ExpectedAttentionPress wrapper
    streaming.py       # StreamingLLMPress wrapper
    snapkv.py          # SnapKVPress + PyramidKVPress wrappers
    modality.py        # ModalityAwarePress (novel)
  modality_mask.py     # build_image_mask(input_ids, processor)
  load_model.py        # (existing) Qwen3-VL loader
  utils.py             # (existing) GPU memory / timing helpers

eval/
  eval_kv_compression.py   # (existing) H2O-only eval, still works
  eval_kv_methods.py       # new, supports --method for all four

scripts/
  run_kv_compression.sh    # single-run launcher
  run_kv_sweep.sh          # full method × ratio × dataset sweep

docs/
  kv_compression_plan.md   # this file
```

---

## Common pitfalls

1. **Mask length mismatch.** `ModalityAwarePress.compress` clips/pads the
   mask to `keys.shape[2]`. If you see unexpected behavior, verify with:

   ```python
   from src.modality_mask import build_image_mask, summarize_mask
   mask = build_image_mask(inputs["input_ids"][0], processor)
   print(summarize_mask(mask))
   ```

   The `image_fraction` should roughly match what you'd expect for that
   benchmark (DocVQA ≫ RealWorldQA).

2. **Compression ratio of 1.0 isn't allowed** (kvpress constraint). Use
   at most 0.99.

3. **Attention backends.** kvpress patches attention functions at import
   time. If you add custom CUDA kernels, load kvpress first.

4. **Short prompts.** SnapKV needs `window_size < prompt_len`. For MMMU on
   short questions, drop `window_size` to 16 or use H2O instead.
