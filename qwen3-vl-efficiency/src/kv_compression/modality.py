"""
Modality-aware KV cache compression (the novel contribution of this project).

Motivation
----------
Prior work (VL-Cache, MBQ) shows that vision tokens and language tokens
behave differently in VLM KV caches:
  * Vision tokens are *redundant* (many image patches carry almost no
    question-relevant signal) and relatively insensitive to compression.
  * Language tokens are *sparse* (few, but each one is often load-bearing)
    and very sensitive to compression.

Uniform policies like H2O, StreamingLLM, and SnapKV use one global budget
for the whole prompt.  They therefore either:
  * (low compression) waste budget on redundant image tokens, or
  * (high compression) accidentally evict a critical text token.

The fix is a two-track policy:
  * Aggressive eviction on image positions   (image_compression_ratio high).
  * Gentle eviction on text positions        (text_compression_ratio low).

Implementation strategy
-----------------------
We subclass kvpress's `ScorerPress`.  For the underlying importance score
we reuse the ExpectedAttentionPress formulation (= H2O, solid default) or
SnapKV's prompt-aware score - pluggable via `inner`.

To decide which positions are image vs text we expect a per-prompt mask to
be set on the press object *before* `model.generate()` is called:

    press = make_modality_aware_press(image_compression_ratio=0.7,
                                      text_compression_ratio=0.2)
    press.set_modality_mask(image_mask_bool_1d)   # True = image position
    with press(model):
        model.generate(**inputs, max_new_tokens=64)

`src/modality_mask.py` provides `build_image_mask(input_ids, processor)` to
build that mask from the tokenizer output (it looks for `<|vision_start|>`
/ `<|vision_end|>` markers used by Qwen3-VL).

Budget accounting
-----------------
Given total prompt length L with Li image tokens and Lt text tokens:

    n_keep_image = round(Li * (1 - image_compression_ratio))
    n_keep_text  = round(Lt * (1 - text_compression_ratio))
    n_keep_total = n_keep_image + n_keep_text
    effective_compression = 1 - n_keep_total / L

We log `effective_compression` at runtime so results across modalities stay
comparable with the uniform baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
from torch import nn

from kvpress import ExpectedAttentionPress, SnapKVPress
from kvpress.presses.scorer_press import ScorerPress


InnerMethod = Literal["h2o", "snapkv"]


@dataclass
class ModalityAwarePress(ScorerPress):
    """
    Modality-stratified KV cache eviction.

    Parameters
    ----------
    image_compression_ratio : float
        Fraction of IMAGE-token KV entries to evict.
    text_compression_ratio : float
        Fraction of TEXT-token KV entries to evict.
    inner : "h2o" | "snapkv"
        Which scorer to use under the hood.
    window_size : int
        Only used when inner == "snapkv".
    """

    # NOTE: ScorerPress also defines compression_ratio; we keep it at 0 and
    # override compress() to do our own budget accounting.  The parent's
    # __post_init__ only enforces 0 <= compression_ratio < 1 so 0.0 is fine.
    compression_ratio: float = 0.0

    image_compression_ratio: float = 0.7
    text_compression_ratio: float = 0.2
    inner: InnerMethod = "h2o"
    window_size: int = 64

    # Populated per-prompt by `set_modality_mask`. Shape: (seq_len,), bool.
    # True at image-token positions, False at text-token positions.
    _image_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    _scorer: Optional[ScorerPress] = field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        for name, r in [
            ("image_compression_ratio", self.image_compression_ratio),
            ("text_compression_ratio", self.text_compression_ratio),
        ]:
            if not 0.0 <= r < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {r}")

        if self.inner == "h2o":
            self._scorer = ExpectedAttentionPress(compression_ratio=0.0)
        elif self.inner == "snapkv":
            self._scorer = SnapKVPress(
                compression_ratio=0.0,
                window_size=self.window_size,
            )
        else:
            raise ValueError(f"Unknown inner scorer: {self.inner}")

    # ------------------------------------------------------------------ API

    def set_modality_mask(self, image_mask: torch.Tensor) -> None:
        """
        Register the per-prompt image mask.  Must be called once per prompt,
        BEFORE entering the `with press(model):` block.

        Parameters
        ----------
        image_mask : torch.Tensor of shape (seq_len,), dtype bool
            True at image-token positions, False at text-token positions.
        """
        if image_mask.dtype != torch.bool:
            image_mask = image_mask.bool()
        if image_mask.dim() != 1:
            raise ValueError(
                f"image_mask must be 1D (seq_len,), got shape {tuple(image_mask.shape)}"
            )
        self._image_mask = image_mask

    def effective_compression_ratio(self, seq_len: Optional[int] = None) -> Optional[float]:
        """Report the global eviction fraction actually used for the current prompt."""
        if self._image_mask is None:
            return None
        if seq_len is None:
            seq_len = int(self._image_mask.numel())
        n_img = int(self._image_mask.sum().item())
        n_txt = seq_len - n_img
        n_keep = (
            round(n_img * (1 - self.image_compression_ratio))
            + round(n_txt * (1 - self.text_compression_ratio))
        )
        return 1.0 - n_keep / max(seq_len, 1)

    # ---------------------------------------------------------------- score

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        # Delegate to the inner scorer (H2O or SnapKV).
        return self._scorer.score(module, hidden_states, keys, values, attentions, kwargs)

    # -------------------------------------------------------------- compress

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ):
        # keys shape: (batch, num_kv_heads, seq_len, head_dim)
        k_len = keys.shape[2]

        if self._image_mask is None:
            # No mask registered -> fall back to a single uniform budget.
            # We use the average of the two ratios to stay close to what the
            # caller asked for.
            avg = 0.5 * (self.image_compression_ratio + self.text_compression_ratio)
            if avg == 0.0:
                return keys, values
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            n_keep = int(k_len * (1 - avg))
            indices = scores.topk(n_keep, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()

        # Align mask length with the current prefill length.
        img_mask = self._image_mask.to(keys.device)
        if img_mask.numel() != k_len:
            # Prompt may be truncated/padded; take the trailing k_len positions
            # (kvpress fires compress() at end of prefill on the full prompt).
            if img_mask.numel() > k_len:
                img_mask = img_mask[-k_len:]
            else:
                pad = torch.zeros(k_len - img_mask.numel(), dtype=torch.bool, device=keys.device)
                img_mask = torch.cat([pad, img_mask], dim=0)

        # Importance scores over all positions (shape: batch, num_kv_heads, seq_len)
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Budgets per modality.
        n_img = int(img_mask.sum().item())
        n_txt = k_len - n_img
        n_keep_img = max(1, round(n_img * (1 - self.image_compression_ratio))) if n_img > 0 else 0
        n_keep_txt = max(1, round(n_txt * (1 - self.text_compression_ratio))) if n_txt > 0 else 0

        # Build modality-restricted score tensors by pushing the "other"
        # modality to -inf so topk selects only within the modality.
        neg_inf = torch.finfo(scores.dtype).min

        img_scores = scores.masked_fill(~img_mask, neg_inf)   # keep image-only scores
        txt_scores = scores.masked_fill(img_mask, neg_inf)    # keep text-only scores

        # Per-modality top-k indices (batch, num_kv_heads, n_keep_*).
        if n_keep_img > 0:
            img_idx = img_scores.topk(n_keep_img, dim=-1).indices
        else:
            img_idx = torch.empty(
                scores.shape[0], scores.shape[1], 0, dtype=torch.long, device=scores.device
            )
        if n_keep_txt > 0:
            txt_idx = txt_scores.topk(n_keep_txt, dim=-1).indices
        else:
            txt_idx = torch.empty(
                scores.shape[0], scores.shape[1], 0, dtype=torch.long, device=scores.device
            )

        # Combine and sort (sorted indices preserve natural position order,
        # which matters for RoPE-aware attention after eviction).
        kept = torch.cat([img_idx, txt_idx], dim=-1)
        kept, _ = kept.sort(dim=-1)

        # Gather keys/values at the retained positions.
        gather_idx = kept.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values


def make_modality_aware_press(
    image_compression_ratio: float = 0.7,
    text_compression_ratio: float = 0.2,
    inner: InnerMethod = "h2o",
    window_size: int = 64,
) -> ModalityAwarePress:
    """
    Build a ModalityAwarePress.  See module docstring for usage.
    """
    return ModalityAwarePress(
        image_compression_ratio=image_compression_ratio,
        text_compression_ratio=text_compression_ratio,
        inner=inner,
        window_size=window_size,
    )
