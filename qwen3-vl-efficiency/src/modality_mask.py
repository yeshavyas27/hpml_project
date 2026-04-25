"""
Build a per-position modality mask for a Qwen3-VL prompt.

Qwen3-VL inserts vision tokens between the special markers
`<|vision_start|>` and `<|vision_end|>` inside the chat template.  The actual
image patches are represented by repeated `<|image_pad|>` tokens in the
input_ids produced by the processor (the model replaces them with the
visual-encoder outputs inside the forward pass, but their positions in
input_ids remain the same for our purposes).

We return a 1D boolean tensor of shape (seq_len,) where True means the
position is an image token and False means it is a text (or special /
system) token.  Positions of `<|vision_start|>` and `<|vision_end|>` themselves
are treated as TEXT - they carry semantic structure, not image content.

Usage
-----
    from src.modality_mask import build_image_mask
    mask = build_image_mask(inputs["input_ids"][0], processor)
    press.set_modality_mask(mask)
"""
from __future__ import annotations

from typing import Iterable

import torch

# Token strings used by the Qwen2/3-VL tokenizer.  These match the public
# Qwen/Qwen3-VL-4B-Instruct checkpoint.  If you ever point this code at a
# different VLM family, just change the strings here.
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"  # same role as image_pad but for video frames


def _token_id(processor, token: str) -> int | None:
    """Return the id for `token` in the processor's tokenizer, or None if missing."""
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ids = tok.convert_tokens_to_ids([token])
    tid = ids[0] if ids else None
    # HF returns the UNK id (often 0) for unknown tokens in some tokenizers;
    # detect by round-tripping.
    if tid is None:
        return None
    back = tok.convert_ids_to_tokens([tid])
    if back and back[0] == token:
        return tid
    return None


def get_image_token_ids(processor) -> set[int]:
    """Resolve the set of input_ids that count as IMAGE tokens for this processor."""
    candidates = [IMAGE_PAD, VIDEO_PAD]
    ids: list[int] = []
    for t in candidates:
        tid = _token_id(processor, t)
        if tid is not None:
            ids.append(tid)
    return set(ids)


def build_image_mask(input_ids: torch.Tensor | Iterable[int], processor) -> torch.Tensor:
    """
    Build a (seq_len,) bool mask where True marks image-token positions.

    Parameters
    ----------
    input_ids : 1D LongTensor of shape (seq_len,) or an iterable of ints.
        The tokenized prompt.  If you have a batched tensor of shape
        (batch, seq_len), pass `input_ids[0]` for a single prompt.
    processor : HF AutoProcessor
        The Qwen3-VL processor returned by `load_model_and_processor()`.

    Returns
    -------
    torch.BoolTensor of shape (seq_len,)
    """
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 2:
            if input_ids.size(0) != 1:
                raise ValueError(
                    "build_image_mask expects a single prompt; pass input_ids[i] "
                    "for prompt i in a batched tensor."
                )
            input_ids = input_ids[0]
        ids_tensor = input_ids.detach().to("cpu").long()
    else:
        ids_tensor = torch.tensor(list(input_ids), dtype=torch.long)

    image_ids = get_image_token_ids(processor)
    if not image_ids:
        # Processor didn't expose image_pad - nothing we can mark, return all False.
        return torch.zeros_like(ids_tensor, dtype=torch.bool)

    mask = torch.zeros_like(ids_tensor, dtype=torch.bool)
    for tid in image_ids:
        mask |= ids_tensor == tid
    return mask


def summarize_mask(mask: torch.Tensor) -> dict:
    """Small helper for logging: returns {n_total, n_image, n_text, image_fraction}."""
    n_total = int(mask.numel())
    n_image = int(mask.sum().item())
    return {
        "n_total": n_total,
        "n_image": n_image,
        "n_text": n_total - n_image,
        "image_fraction": (n_image / n_total) if n_total else 0.0,
    }
