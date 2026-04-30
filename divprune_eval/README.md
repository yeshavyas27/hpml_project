# HPML Project — DivPrune: Visual Token Pruning for Efficient Multimodal Inference

## What is DivPrune?

DivPrune is a training-free visual token pruning method for vision-language models. Given a high-resolution image tokenized into a large number of visual patch tokens, DivPrune selects a diverse, representative subset of those tokens before they are fed into the language model — reducing the effective sequence length and, in principle, the cost of attention and decoding.

The core idea: instead of keeping all image tokens, DivPrune scores tokens by their diversity relative to each other (e.g., via clustering or distance-based selection) and retains only `subset_ratio * total_image_tokens` tokens. The rest are dropped. This is applied at inference time with no fine-tuning required.

We evaluate DivPrune on four VQA benchmarks — **DocVQA**, **MathVista**, **MMMU**, and **RealWorldQA** — using the **Qwen3-VL** model, sweeping retention ratios of `r ∈ {0.2, 0.3, 0.5}` against a no-pruning baseline.

---

## Raw Results

Averaged over all samples per file. `baseline` = no pruning applied.

| Dataset | Ratio | N | Accuracy | Prefill (ms) | Decode (ms) | Tokens Gen | Throughput (tok/s) | Peak GPU Mem (GB) | Total Img Tokens | Kept Img Tokens | Orig Seq Len | Pruned Seq Len | Retention Ratio | Seq Reduction Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| docvqa | baseline | 100 | 0.7700 | 108.54 | 978.30 | 19.20 | 19.40 | 8.5003 | N/A | N/A | N/A | N/A | N/A | N/A |
| docvqa | 0.2 | 100 | 0.3800 | 152.77 | 1410.45 | 27.60 | 19.49 | 8.4260 | 761.28 | 152.20 | 783.54 | 174.46 | 0.1999 | 0.2227 |
| docvqa | 0.3 | 100 | 0.5000 | 170.67 | 1363.41 | 25.82 | 18.82 | 8.4260 | 761.28 | 228.44 | 783.54 | 250.70 | 0.3001 | 0.3200 |
| docvqa | 0.5 | 100 | 0.6400 | 201.45 | 1298.11 | 24.70 | 18.90 | 8.4260 | 761.28 | 380.64 | 783.54 | 402.90 | 0.5000 | 0.5142 |
| mathvista | baseline | 100 | 1.0000 | 158.65 | 1045.30 | 19.86 | 18.58 | 8.4278 | N/A | N/A | N/A | N/A | N/A | N/A |
| mathvista | 0.2 | 100 | 1.0000 | 166.68 | 1180.76 | 23.22 | 19.37 | 8.3810 | 502.39 | 100.40 | 579.66 | 177.67 | 0.1998 | 0.3065 |
| mathvista | 0.3 | 100 | 1.0000 | 178.10 | 1113.13 | 22.30 | 19.66 | 8.3820 | 502.39 | 150.79 | 579.66 | 228.06 | 0.3001 | 0.3934 |
| mathvista | 0.5 | 100 | 1.0000 | 208.29 | 1148.58 | 22.60 | 19.31 | 8.3865 | 502.39 | 251.16 | 579.66 | 328.43 | 0.4999 | 0.5666 |
| mmmu | baseline | 29 | 0.0690 | 311.08 | 479.78 | 8.97 | 18.37 | 8.3921 | N/A | N/A | N/A | N/A | N/A | N/A |
| mmmu | 0.2 | 29 | 0.0000 | 143.00 | 457.97 | 8.62 | 18.47 | 8.3543 | 251.86 | 50.38 | 459.48 | 258.00 | 0.2000 | 0.5615 |
| mmmu | 0.3 | 29 | 0.0345 | 145.52 | 447.75 | 8.52 | 18.68 | 8.3580 | 251.86 | 75.59 | 459.48 | 283.21 | 0.3001 | 0.6164 |
| mmmu | 0.5 | 29 | 0.0000 | 177.51 | 456.78 | 8.62 | 18.48 | 8.3666 | 251.86 | 125.90 | 459.48 | 333.52 | 0.4999 | 0.7259 |
| realworldqa | baseline | 100 | 0.7100 | 165.37 | 55.33 | 1.00 | 18.14 | 8.6376 | N/A | N/A | N/A | N/A | N/A | N/A |
| realworldqa | 0.2 | 100 | 0.6900 | 196.88 | 53.79 | 1.00 | 18.65 | 8.5321 | 1306.61 | 261.42 | 1355.54 | 310.35 | 0.2001 | 0.2289 |
| realworldqa | 0.3 | 100 | 0.7200 | 227.54 | 55.68 | 1.00 | 18.03 | 8.5321 | 1306.61 | 392.21 | 1355.54 | 441.14 | 0.3002 | 0.3254 |
| realworldqa | 0.5 | 100 | 0.7000 | 270.21 | 53.99 | 1.00 | 18.60 | 8.5327 | 1306.61 | 652.98 | 1355.54 | 701.91 | 0.4998 | 0.5178 |

---

## Averaged Across Datasets (per Approach)

Macro-average over all four datasets. MMMU (n=29) is included as-is; its low sample count should be noted when interpreting accuracy.

| Approach | Accuracy | Prefill (ms) | Decode (ms) | Tokens Gen | Throughput (tok/s) | Peak GPU Mem (GB) | Kept Img Tokens | Pruned Seq Len | Retention Ratio | Seq Reduction Ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.6373 | 185.91 | 639.68 | 12.26 | 18.62 | 8.4895 | N/A | N/A | N/A | N/A |
| r=0.2 | 0.5175 | 164.83 | 775.74 | 15.11 | 18.99 | 8.4234 | 141.10 | 230.12 | 0.1999 | 0.3299 |
| r=0.3 | 0.5636 | 180.46 | 744.99 | 14.41 | 18.80 | 8.4245 | 211.76 | 300.78 | 0.3001 | 0.4138 |
| r=0.5 | 0.5850 | 214.37 | 739.37 | 14.23 | 18.82 | 8.4280 | 352.67 | 441.69 | 0.5000 | 0.5811 |

---

## Analysis

### Accuracy by Dataset and Retention Ratio

| Dataset | Baseline | r=0.2 | r=0.3 | r=0.5 |
|---|---|---|---|---|
| **DocVQA** | 0.770 | 0.380 (-51%) | 0.500 (-35%) | 0.640 (-17%) |
| **MathVista** | 1.000 | 1.000 (0%) | 1.000 (0%) | 1.000 (0%) |
| **MMMU** | 0.069 | 0.000 (-100%) | 0.035 (-50%) | 0.000 (-100%) |
| **RealWorldQA** | 0.710 | 0.690 (-3%) | 0.720 (+1%) | 0.700 (-1%) |

### Prefill Latency (ms) — DivPrune vs Baseline

| Dataset | Baseline | r=0.2 | r=0.3 | r=0.5 |
|---|---|---|---|---|
| **DocVQA** | 108.5 | 152.8 (+41%) | 170.7 (+57%) | 201.4 (+86%) |
| **MathVista** | 158.6 | 166.7 (+5%) | 178.1 (+12%) | 208.3 (+31%) |
| **MMMU** | 311.1 | 143.0 (-54%) | 145.5 (-53%) | 177.5 (-43%) |
| **RealWorldQA** | 165.4 | 196.9 (+19%) | 227.5 (+38%) | 270.2 (+63%) |

### Key Observations

| Observation | Detail |
|---|---|
| **Accuracy** | MathVista: no degradation at any ratio. RealWorldQA: stable (~70%). DocVQA: heavy loss at r=0.2 (-51%), recovers at r=0.5 (-17%). MMMU: unreliable (n=29, near-zero baseline). |
| **Prefill latency** | Increases with pruning on 3/4 datasets — DivPrune overhead exceeds attention savings. MMMU uniquely benefits (-54%) due to very long sequences. |
| **Decode latency** | Mostly unaffected except DocVQA, where pruning increases decode time (+44% at r=0.2) — likely due to more generated tokens when context is degraded. |
| **Tokens generated** | DocVQA generates more tokens under pruning (19→28), suggesting the model produces more verbose answers when visual context is incomplete. |
| **Throughput** | Flat across all conditions (~18–20 tok/s) — pruning yields no measurable throughput gain at these scales. |
| **GPU memory** | Negligible savings (<0.2 GB across all datasets and ratios). |
| **Seq reduction vs retention** | High-res image datasets (RealWorldQA, DocVQA) show seq_reduction ≈ retention_ratio. Text-heavy datasets (MathVista, MMMU) show seq_reduction > retention_ratio because text tokens form a large non-prunable floor, limiting effective sequence reduction. |
| **Averaged accuracy drop** | Macro-averaged accuracy falls from 0.637 (baseline) to 0.518 at r=0.2, recovering partially to 0.585 at r=0.5 — a non-trivial degradation at all pruning levels. |

---

## Column Definitions

- **Ratio**: DivPrune image-token retention ratio (`subset_ratio`); `baseline` = no pruning applied
- **N**: Number of samples evaluated
- **Accuracy**: Fraction of samples answered correctly
- **Prefill (ms)**: Average prefill latency in milliseconds
- **Decode (ms)**: Average decode latency in milliseconds
- **Tokens Gen**: Average number of tokens generated per sample
- **Throughput (tok/s)**: Average generation throughput in tokens per second
- **Peak GPU Mem (GB)**: Average peak GPU memory usage in GB
- **Total Img Tokens**: Average total image tokens before DivPrune pruning
- **Kept Img Tokens**: Average image tokens retained after pruning
- **Orig Seq Len**: Average sequence length before pruning
- **Pruned Seq Len**: Average sequence length after pruning
- **Retention Ratio**: `kept_image_tokens / total_image_tokens`
- **Seq Reduction Ratio**: `pruned_seq_len / original_seq_len`
