# Qwen3-VL Efficiency

This project studies inference efficiency for **Qwen3-VL-4B-Instruct**, a multimodal vision-language model.

The goal is to understand and improve:
- latency (speed)
- GPU memory usage
- throughput

We begin by running **baseline (no optimization)** experiments and later apply techniques like:
- visual token reduction
- KV cache compression
- modality-aware optimization

---

## Baseline Results (no optimization)

| Dataset     | Accuracy | Prefill (ms) | Decode (ms) | Memory (GB) | Throughput (tok/s) |
|-------------|----------|-------------|-------------|-------------|-------------------|
| MathVista   | 35.9%    | 222.6       | 296.6       | 9.03        | 12.8              |
| MMMU        | 46.4%    | 216.4       | 174.4       | 9.05        | 11.4              |
| RealWorldQA | 71.6%    | 456.3       | 69.7        | 9.27        | 4.2               |
| DocVQA      | 87.8%    | 1649.3      | 279.3       | 9.94        | 3.8               |

Prefill latency scales with image token count. DocVQA is most expensive (3600+ image tokens per sample).

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Phase 1: Baseline Evaluation

Evaluates the model with no optimizations to establish reference performance.

### Benchmarks

| Script | Dataset | Task |
|--------|---------|------|
| `eval/eval_realworldqa.py` | xai-org/RealworldQA | General visual reasoning |
| `eval/eval_mathvista.py`   | AI4Math/MathVista   | Visual math reasoning    |
| `eval/eval_mmmu.py`        | MMMU/MMMU (Math)    | Multiple-choice multimodal |
| `eval/eval_docvqa.py`      | HuggingFaceM4/DocumentVQA | Document understanding |

### Run baseline

```bash
bash scripts/run_realworldqa.sh
bash scripts/run_mathvista.sh
bash scripts/run_mmmu.sh
bash scripts/run_docvqa.sh
```

Results are saved to `results/baseline/`.

---

## Milestone 1: KV Cache Compression (H2O)

### Background

In autoregressive decoding the model stores **Key** and **Value** tensors for
every input token across all transformer layers — the **KV cache**. For
multimodal inputs like DocVQA (3600+ image tokens) this cache is large,
consumes significant GPU memory, and must be re-read at every decode step,
increasing latency.

**H2O (Heavy-Hitter Oracle)** is a training-free compression method. During
the prefill phase it accumulates the average attention weight each token
receives across all heads. After prefill it evicts the tokens with the
lowest cumulative attention scores — the "non-heavy-hitters". Only the
top-k most-attended tokens remain in the KV cache for decoding.

In `kvpress` this method is called `ExpectedAttentionPress`.

### Key concept: compression_ratio

| `compression_ratio` | Effect |
|---------------------|--------|
| `0.0` | No compression — keep 100% of tokens (baseline parity) |
| `0.3` | Evict 30% of least-attended tokens, keep 70% |
| `0.5` | Evict 50% of tokens, keep 50% (recommended starting point) |
| `0.7` | Evict 70% of tokens, keep 30% (aggressive, may hurt accuracy) |

### How it is implemented

#### 1. `src/kv_compression.py` — press factory

```python
from src.kv_compression import make_h2o_press

press = make_h2o_press(compression_ratio=0.5)
```

`make_h2o_press` validates the ratio and returns an `ExpectedAttentionPress`
instance ready to be used as a context manager.

#### 2. `eval/eval_kv_compression.py` — evaluation script

Applies the press around every `model.generate()` call:

```python
with press(model):
    prediction, prefill_ms, decode_ms, num_tokens, throughput = (
        generate_with_timing(model, processor, image, question, max_new_tokens)
    )
```

The context manager registers attention hooks on the model for the duration
of the `with` block. When `model.generate()` runs inside the block:
1. Prefill: attention weights are accumulated per token.
2. After prefill: bottom `compression_ratio` fraction of tokens are evicted.
3. Decode: only the retained heavy-hitter tokens are in the KV cache.

Timing is measured using `TextIteratorStreamer` in a background thread so
prefill and decode phases can be timed separately without modifying model
internals.

Metrics collected per sample:

| Field | Description |
|-------|-------------|
| `prefill_ms` | Time from call start to first generated token (ms) |
| `decode_ms` | Time from first token to last token (ms) |
| `num_tokens_generated` | Count of new tokens produced |
| `throughput_tok_per_sec` | Tokens per second during decode |
| `peak_gpu_mem_gb` | Peak GPU memory in GB (via `torch.cuda.max_memory_allocated`) |
| `correct` | Approximate accuracy flag (see accuracy notes below) |

### Step-by-step: run KV compression evaluation

**Step 1.** Install dependencies (includes `kvpress`):

```bash
pip install -r requirements.txt
```

**Step 2.** Run with the default script (RealWorldQA, ratio=0.5, 5 samples):

```bash
bash scripts/run_kv_compression.sh
```

**Step 3.** Run on a specific dataset and ratio:

```bash
python -m eval.eval_kv_compression \
    --dataset realworldqa \
    --compression_ratio 0.5 \
    --max_samples 5
```

**Step 4.** Sweep multiple compression ratios to find the accuracy/efficiency trade-off:

```bash
for ratio in 0.0 0.3 0.5 0.7; do
    python -m eval.eval_kv_compression \
        --dataset realworldqa \
        --compression_ratio $ratio \
        --max_samples 50
done
```

**Step 5.** Run across all four benchmarks:

```bash
for dataset in realworldqa mathvista mmmu docvqa; do
    python -m eval.eval_kv_compression \
        --dataset $dataset \
        --compression_ratio 0.5 \
        --max_samples 50
done
```

Results are written to `results/kv_compression/` as JSONL files:

```
results/kv_compression/
  realworldqa_cr0p50.jsonl
  mathvista_cr0p50.jsonl
  mmmu_cr0p50.jsonl
  docvqa_cr0p50.jsonl
```

**Step 6.** Override via environment variables (quick one-liners):

```bash
DATASET=docvqa RATIO=0.7 MAX_SAMPLES=10 bash scripts/run_kv_compression.sh
```

### Accuracy notes

The `correct` flag in each result row uses an approximate matching heuristic:

- **Multiple-choice** (mmmu, realworldqa): checks whether the first character
  of the prediction matches the expected answer letter (A/B/C/D).
- **DocVQA**: accepts the prediction if it contains (or is contained by) any
  of the valid ground-truth answer strings.
- **MathVista**: case-insensitive substring match against the expected answer.

For publication-quality numbers, replace this with task-specific metrics
(ANLS for DocVQA, exact-match for MathVista, etc.).

### Expected outcomes

At `compression_ratio=0.5` we expect:
- Memory reduction of ~20–40% for image-heavy benchmarks.
- Prefill time roughly unchanged (compression happens at the end of prefill).
- Decode time reduced (smaller KV cache → faster attention per decode step).
- Accuracy drop of 1–5 percentage points depending on benchmark.

DocVQA should show the largest efficiency gain because it has the most image
tokens (3600+), giving H2O the most tokens to evict.

---

## Project structure

```
qwen3-vl-efficiency/
├── requirements.txt
├── src/
│   ├── load_model.py        # loads Qwen3-VL-4B-Instruct
│   ├── utils.py             # GPU memory / timing helpers
│   └── kv_compression.py   # H2O press factory (Milestone 1)
├── eval/
│   ├── eval_realworldqa.py  # baseline
│   ├── eval_mathvista.py    # baseline
│   ├── eval_mmmu.py         # baseline
│   ├── eval_docvqa.py       # baseline
│   └── eval_kv_compression.py  # Milestone 1: KV compression
├── scripts/
│   ├── run_realworldqa.sh
│   ├── run_mathvista.sh
│   ├── run_mmmu.sh
│   ├── run_docvqa.sh
│   └── run_kv_compression.sh   # Milestone 1 launcher
└── results/
    ├── baseline/            # baseline JSONL outputs
    └── kv_compression/      # Milestone 1 outputs
```
