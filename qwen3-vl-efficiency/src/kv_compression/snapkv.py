"""
SnapKV and PyramidKV wrappers.

SnapKV (https://arxiv.org/abs/2404.14469)
-----------------------------------------
SnapKV is "prompt-aware" eviction: instead of scoring every prompt token by
how much generic attention it is expected to receive, SnapKV uses the
attention pattern of the LAST `window_size` query positions of the prompt
(typically the user's actual question) as a probe.  A key is kept if the
question's queries attend strongly to it; otherwise it is evicted.

This is a good fit for VQA-style tasks where the question explicitly selects
which parts of the image matter - e.g. "what is the number on the left axis?"
attends to the axis region, not the full chart.

PyramidKV (https://arxiv.org/abs/2406.02069)
--------------------------------------------
Same SnapKV scoring, but the KV budget is not uniform across layers.
Lower layers get more budget (they need more context), higher layers get
less.  `beta` controls how steep the pyramid is.  For VLMs this is worth
testing because Qwen3-VL's DeepStack fuses multiple visual-encoder layers
into the LM input, so the "how much visual context does layer L really
need" question has a less obvious answer than in text-only LLMs.
"""
from kvpress import SnapKVPress, PyramidKVPress


def make_snapkv_press(
    compression_ratio: float = 0.5,
    window_size: int = 64,
    kernel_size: int = 5,
):
    """
    Parameters
    ----------
    compression_ratio : float
        Fraction of prompt KV entries to evict.
    window_size : int
        Number of final prompt tokens whose attention is used as the scorer.
        64 is the paper default.  Must be <= prompt length minus a few.
    kernel_size : int
        1D max-pool kernel over the attention scores (smoothing, default 5).
    """
    if not 0.0 <= compression_ratio < 1.0:
        raise ValueError(f"compression_ratio must be in [0, 1), got {compression_ratio}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    return SnapKVPress(
        compression_ratio=compression_ratio,
        window_size=window_size,
        kernel_size=kernel_size,
    )


def make_pyramid_press(
    compression_ratio: float = 0.5,
    window_size: int = 64,
    kernel_size: int = 5,
    beta: int = 20,
):
    """
    PyramidKV = SnapKV scoring + per-layer pyramid budget.

    Parameters
    ----------
    beta : int
        Steepness of the pyramid.  Larger beta -> later layers get much
        smaller budget than earlier layers.  The kvpress default is 20.
    """
    if not 0.0 <= compression_ratio < 1.0:
        raise ValueError(f"compression_ratio must be in [0, 1), got {compression_ratio}")
    if beta < 1:
        raise ValueError(f"beta must be >= 1, got {beta}")
    return PyramidKVPress(
        compression_ratio=compression_ratio,
        window_size=window_size,
        kernel_size=kernel_size,
        beta=beta,
    )
