"""
H2O-style KV cache compression.

We use kvpress's `ExpectedAttentionPress`, which is a statistically principled
version of the original H2O "heavy-hitter" method.  Instead of accumulating
observed attention weights during prefill (which is noisy and head-dependent),
it models query statistics analytically and computes the expected attention
each key position will receive from future queries, then evicts the positions
with the lowest expected attention.

Why this matters for Qwen3-VL
-----------------------------
Image tokens make up the bulk of the prompt on document- and chart-heavy
benchmarks (DocVQA, MathVista, MMMU).  H2O is our baseline for answering:
"does a single global attention score suffice, or do image tokens need a
different eviction policy?"  The modality-aware press in this package lets us
answer that by reusing the same expected-attention scorer.
"""
from kvpress import ExpectedAttentionPress


def make_h2o_press(compression_ratio: float = 0.5):
    """
    Build an H2O-style press.

    Parameters
    ----------
    compression_ratio : float
        Fraction of KV cache entries to EVICT.
        0.0 -> keep everything (baseline parity).
        0.5 -> evict 50%, keep 50% (good default).
        0.7 -> aggressive, accuracy usually starts dropping here.

    Returns
    -------
    kvpress.ExpectedAttentionPress
        Use as a context manager:   with press(model): model.generate(...)
    """
    if not 0.0 <= compression_ratio < 1.0:
        raise ValueError(
            f"compression_ratio must be in [0, 1), got {compression_ratio}"
        )
    return ExpectedAttentionPress(compression_ratio=compression_ratio)
