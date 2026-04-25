"""
StreamingLLM-style KV cache compression.

Paper: https://arxiv.org/abs/2309.17453

The idea is extremely simple:
  - Always keep the first `n_sink` tokens ("attention sinks"): empirically,
    LLMs route a large fraction of their attention mass to the very first
    few tokens regardless of content, and evicting them collapses quality.
  - Keep the most recent N tokens (the "rolling window").
  - Evict everything in the middle.

For Qwen3-VL this is the harshest policy in our suite because image tokens
live in the middle of the prompt (between the system prompt and the question),
so StreamingLLM will indiscriminately throw most of them away.  Useful as the
"worst reasonable method" line on our memory / accuracy plots: if modality-
aware compression can't beat this, it's not working.
"""
from kvpress import StreamingLLMPress


def make_streaming_press(compression_ratio: float = 0.5, n_sink: int = 4):
    """
    Parameters
    ----------
    compression_ratio : float
        Fraction of KV cache entries to evict (middle portion).
    n_sink : int
        Number of leading tokens to always keep. 4 matches the paper.
    """
    if not 0.0 <= compression_ratio < 1.0:
        raise ValueError(
            f"compression_ratio must be in [0, 1), got {compression_ratio}"
        )
    if n_sink < 1:
        raise ValueError(f"n_sink must be >= 1, got {n_sink}")
    return StreamingLLMPress(
        compression_ratio=compression_ratio,
        n_sink=n_sink,
    )
