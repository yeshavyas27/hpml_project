"""
DivPrune visual token pruning for Qwen3-VL (HuggingFace Transformers >= 4.57).

=== DESIGN ===

DivPrune greedily selects a diverse subset of visual tokens using pairwise
cosine distance (max-min selection). Unlike FastV (which prunes the KV cache
after layer K based on attention importance), DivPrune prunes visual tokens
BEFORE the first LLM layer, maximising feature coverage.

Hook point: register_forward_pre_hook on language_model.layers[0].
  - args[0] = hidden_states [B, seq_len, hidden_dim]: merged visual+text embeds
    output by Qwen3VLModel (after ViT, merger, deepstack, embed_tokens)
  - The hook extracts image token embeddings, runs DivPrune greedy selection,
    removes non-selected image tokens from hidden_states, position_ids,
    position_embeddings (cos/sin), and attention_mask, then returns the
    shortened tensors as (new_args, new_kwargs).
  - All 36 LLM layers then process the pruned sequence natively.

=== DivPrune Algorithm ===

Given N image token embeddings F[0..N-1] of dimension D:
  1. Compute pairwise cosine-distance matrix D_ij = 1 - cos_sim(F_i, F_j)
  2. Greedily select k = round(r * N) diverse tokens:
     - Step 0: pick token maximally distant from all others
       (highest second-smallest distance — excludes self-dist=0)
     - Step i>0: pick token with maximum min-distance to any already-selected token
  3. Keep selected tokens + all non-image tokens; discard the rest.

Reference: DivPrune (Hamid Kazemi et al.), originally for LLaVA-1.5/1.6.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Config & Stats
# ---------------------------------------------------------------------------

@dataclass
class DivPruneConfig:
    """Configuration for DivPrune visual token pruning."""
    use_divprune: bool = True
    divprune_r: float = 0.5        # fraction of image tokens to KEEP
    image_token_id: int = 151655   # <|image_pad|> token ID for Qwen3-VL
    verbose: bool = False


@dataclass
class DivPruneStats:
    """Per-sample statistics. Returned by get_divprune_stats(model)."""
    pruning_applied: bool = False
    total_image_tokens: int = 0
    kept_image_tokens: int = 0
    pruned_image_tokens: int = 0
    original_seq_len: int = 0
    pruned_seq_len: int = 0
    keep_ratio: float = 1.0

    @property
    def actual_keep_ratio(self) -> float:
        if self.total_image_tokens == 0:
            return 1.0
        return self.kept_image_tokens / self.total_image_tokens

    @property
    def compression_ratio(self) -> float:
        if self.original_seq_len == 0:
            return 1.0
        return self.pruned_seq_len / self.original_seq_len


class _DivPruneState:
    def __init__(self, config: DivPruneConfig):
        self.config = config
        self.image_token_positions: Optional[torch.Tensor] = None
        self.pruning_done: bool = False
        self.stats: DivPruneStats = DivPruneStats()

    def reset(self):
        self.image_token_positions = None
        self.pruning_done = False
        self.stats = DivPruneStats()


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    """
    Pairwise cosine DISTANCE (1 - cosine_similarity).
    x: [N, D]  ->  returns [N, N]
    """
    xn = F.normalize(x.float(), p=2, dim=-1)
    return 1.0 - (xn @ xn.T)


def _divprune_select(features: torch.Tensor, n_keep: int) -> torch.Tensor:
    """
    Greedy max-min diversity selection.

    features : [N, D]  — image token embeddings
    n_keep   : int     — number of tokens to keep

    Returns [n_keep] indices (into the N image tokens) of selected tokens.
    """
    N = features.shape[0]
    if n_keep >= N:
        return torch.arange(N, device=features.device)

    dist = _pairwise_cosine_distance(features)  # [N, N]
    selected = torch.empty(n_keep, dtype=torch.long, device=features.device)

    for i in range(n_keep):
        if i == 0:
            # Pick most isolated token: highest second-smallest row distance.
            # (smallest is always 0 = self; second-smallest is nearest neighbour)
            scores = torch.topk(dist, 2, dim=0, largest=False).values[1, :]
        else:
            # Min distance to any already-selected token
            m2 = dist[selected[:i], :]  # [i, N]
            scores = torch.min(m2, dim=0).values

        selected[i] = torch.argmax(scores)

    return selected


# ---------------------------------------------------------------------------
# Pre-layer-0 hook
# ---------------------------------------------------------------------------

def _make_pre_layer0_hook(state: _DivPruneState):
    """
    Returns a register_forward_pre_hook callable for language_model.layers[0].

    On the prefill pass, prunes image tokens from:
      hidden_states   (args[0])
      position_ids    (kwargs, [4,B,S] MRoPE or [B,S])
      position_embeddings  (kwargs, (cos,sin) tuple)
      attention_mask  (kwargs, 2-D or 4-D causal mask)
    """
    def _hook(module, args, kwargs):
        # Only fire once per generate() call, and only on prefill
        if state.pruning_done or state.image_token_positions is None:
            return args, kwargs

        if not args or not isinstance(args[0], torch.Tensor):
            return args, kwargs

        hidden_states = args[0]                         # [B, S, D]
        B, seq_len, hidden_dim = hidden_states.shape

        image_mask = state.image_token_positions        # [B, orig_S]
        if image_mask.shape[1] > seq_len:
            image_mask = image_mask[:, :seq_len]
        elif image_mask.shape[1] < seq_len:
            # decode step — skip
            return args, kwargs

        if not image_mask.any():
            state.pruning_done = True
            return args, kwargs

        # ── Single-batch fast path (standard eval loop) ──────────────────
        if B == 1:
            img_pos = torch.where(image_mask[0, :seq_len])[0]  # [N_img]
            n_img = img_pos.numel()
            n_keep = max(1, int(round(n_img * state.config.divprune_r)))

            # DivPrune greedy selection
            img_feats = hidden_states[0, img_pos]               # [N_img, D]
            sel_local = _divprune_select(img_feats, n_keep)     # [n_keep]
            sel_pos = img_pos[sel_local]                        # global positions

            # keep_mask: True = keep this position
            keep_mask = (~image_mask[0, :seq_len]).clone()      # all non-image = True
            keep_mask[sel_pos] = True
            kept_idx = torch.where(keep_mask)[0]                # [new_S]
            new_S = kept_idx.numel()

            # ── Prune hidden_states ──────────────────────────────────────
            new_hs = hidden_states[:, kept_idx, :]
            new_args = (new_hs,) + args[1:]

            new_kwargs = dict(kwargs)

            # ── Prune position_ids ───────────────────────────────────────
            pid = kwargs.get("position_ids")
            if pid is not None:
                try:
                    if pid.dim() == 3 and pid.shape[-1] == seq_len:
                        # M-RoPE: [4, B, S]
                        new_kwargs["position_ids"] = pid[:, :, kept_idx]
                    elif pid.dim() == 2 and pid.shape[-1] == seq_len:
                        new_kwargs["position_ids"] = pid[:, kept_idx]
                except Exception:
                    pass

            # ── Prune position_embeddings (cos, sin) ────────────────────
            pe = kwargs.get("position_embeddings")
            if pe is not None:
                try:
                    cos, sin = pe
                    # Typical shapes: [B, S, head_dim//2] or [1, S, head_dim//2]
                    if cos.shape[-2] == seq_len:
                        new_kwargs["position_embeddings"] = (
                            cos[..., kept_idx, :],
                            sin[..., kept_idx, :],
                        )
                except Exception:
                    pass

            # ── Prune attention_mask ─────────────────────────────────────
            att = kwargs.get("attention_mask")
            if att is not None:
                try:
                    if att.dim() == 4 and att.shape[-1] == seq_len and att.shape[-2] == seq_len:
                        # causal mask [B,1,S,S]
                        new_kwargs["attention_mask"] = att[:, :, kept_idx, :][:, :, :, kept_idx]
                    elif att.dim() == 4 and att.shape[-1] == seq_len:
                        new_kwargs["attention_mask"] = att[:, :, :, kept_idx]
                    elif att.dim() == 2 and att.shape[-1] == seq_len:
                        new_kwargs["attention_mask"] = att[:, kept_idx]
                except Exception:
                    pass

            # ── Record stats ─────────────────────────────────────────────
            state.stats = DivPruneStats(
                pruning_applied=True,
                total_image_tokens=n_img,
                kept_image_tokens=n_keep,
                pruned_image_tokens=n_img - n_keep,
                original_seq_len=seq_len,
                pruned_seq_len=new_S,
                keep_ratio=state.config.divprune_r,
            )
            state.pruning_done = True

            if state.config.verbose:
                s = state.stats
                print(
                    f"[DivPrune] ✓ image: {n_img} → {n_keep} "
                    f"(pruned {n_img - n_keep}, {s.actual_keep_ratio:.0%} kept)  |  "
                    f"seq: {seq_len} → {new_S} "
                    f"({s.compression_ratio:.1%})"
                )

            return new_args, new_kwargs

        # ── Multi-batch fallback (skip pruning) ──────────────────────────
        state.pruning_done = True
        return args, kwargs

    return _hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_divprune_to_qwen(model, config: DivPruneConfig):
    """
    Apply DivPrune to a Qwen3VLForConditionalGeneration model.

    Non-invasive: does NOT patch any forward() method.

    Changes:
      - language_model.layers[0] : pre-forward hook (with_kwargs=True)
      - model.generate            : wrapped to reset state + capture input_ids
      - model._divprune_*         : bookkeeping attributes

    Returns the same model, patched in-place.
    """
    if not config.use_divprune:
        return model

    if hasattr(model, "_divprune_hook_handle"):
        print("[DivPrune] WARNING: already applied — call remove_divprune() first.")
        return model

    lang_model = model.model.language_model
    layers = lang_model.layers

    state = _DivPruneState(config)

    hook_handle = layers[0].register_forward_pre_hook(
        _make_pre_layer0_hook(state),
        with_kwargs=True,
    )

    original_generate = model.generate

    def wrapped_generate(*args, **kwargs):
        state.reset()
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args and isinstance(args[0], torch.Tensor):
            input_ids = args[0]
        if input_ids is not None:
            state.image_token_positions = (
                input_ids == config.image_token_id
            ).to(input_ids.device)
        return original_generate(*args, **kwargs)

    model.generate = wrapped_generate
    model._divprune_hook_handle = hook_handle
    model._divprune_state = state
    model._divprune_original_generate = original_generate

    print(
        f"[DivPrune] ✓ Applied to Qwen3-VL  |  "
        f"keep_ratio={config.divprune_r:.0%}  "
        f"image_token_id={config.image_token_id}"
    )
    if config.verbose:
        print("[DivPrune]   verbose=True — will print per-sample pruning info")
    else:
        print("[DivPrune]   Set verbose=True in DivPruneConfig to confirm pruning fires")

    return model


def remove_divprune(model) -> None:
    """Remove DivPrune hook and restore model.generate."""
    if not hasattr(model, "_divprune_hook_handle"):
        print("[DivPrune] Nothing to remove.")
        return
    model._divprune_hook_handle.remove()
    model.generate = model._divprune_original_generate
    del model._divprune_hook_handle
    del model._divprune_state
    del model._divprune_original_generate
    print("[DivPrune] Removed — model.generate restored to original.")


def get_divprune_stats(model) -> DivPruneStats:
    """Returns DivPruneStats for the most recent generate() call."""
    state = getattr(model, "_divprune_state", None)
    if state is None:
        return DivPruneStats()
    return state.stats


def is_divprune_active(model) -> bool:
    """True if DivPrune is currently applied."""
    return hasattr(model, "_divprune_hook_handle")


def get_divprune_config(model) -> Optional[DivPruneConfig]:
    """Returns active DivPruneConfig, or None."""
    state = getattr(model, "_divprune_state", None)
    return state.config if state is not None else None
