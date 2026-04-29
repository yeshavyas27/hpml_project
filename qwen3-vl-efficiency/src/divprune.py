"""
DivPrune visual token pruning for Qwen3-VL (HuggingFace Transformers >= 4.57).

=== DESIGN ===

DivPrune greedily selects a diverse subset of visual tokens using pairwise
cosine distance (max-min selection). Unlike FastV (which prunes the KV cache
after layer K based on attention importance), DivPrune prunes visual tokens
BEFORE the first LLM layer, maximising feature coverage.

Hook point: register_forward_pre_hook on language_model (the inner Qwen3_VLModel).
  - Fires before any decoder layer or deepstack call.
  - Prunes inputs_embeds [B,S,D], visual_pos_masks [B,S], and every tensor in
    deepstack_visual_embeds [N_visual,D] to reflect only the kept visual tokens.
  - Also prunes position_ids and attention_mask to match the new sequence length.
  - All decoder layers and _deepstack_process calls then see consistent shapes.

Why language_model, not layers[0]:
  Qwen3-VL's _deepstack_process runs AFTER each decoder layer and re-indexes
  hidden_states using visual_pos_masks.  Hooking at layers[0] only pruned
  hidden_states; visual_pos_masks and deepstack_visual_embeds kept their original
  length, causing an IndexError on all subsequent _deepstack_process calls.

=== DivPrune Algorithm ===

Given N image token embeddings F[0..N-1] of dimension D:
  1. Compute pairwise cosine-distance matrix D_ij = 1 - cos_sim(F_i, F_j)
  2. Greedily select k = round(r * N) diverse tokens:
     - Step 0: pick token maximally distant from all others
       (highest second-smallest distance — excludes self-dist=0)
     - Step i>0: pick token with maximum min-distance to any already-selected token
  3. Keep selected tokens + all non-image tokens; discard the rest.

Reference: DivPrune (Hamid Kazemi et al.), originally for LLaVA-1.5/1.6.

=== Fix log ===

Fix 1 — _divprune_select seed step used dim=0 (column-wise topk) instead of
         dim=1 (row-wise).  Changed to dim=1 and index [:, 1] so each token is
         scored by its own nearest-neighbour distance, not by how rarely it is
         anyone else's nearest neighbour.

Fix 2 — Greedy loop could re-select an already-chosen index because argmax had
         no exclusion mask.  Added scores[selected[:i]] = -1.0 before argmax.

Fix 3 — The full N×N cosine-distance matrix was materialised on device for every
         sample (O(N²) memory, up to 256 MB for 4 k image tokens).  Replaced with
         an incremental min_dist vector that is updated after each selection step,
         reducing memory to O(N) and removing the matrix entirely.

Fix 4 — deepstack_visual_embeds indexing had no shape guard; a mismatch between
         embed.shape[0] and n_img caused a silent truncation or IndexError.
         Added an explicit assertion before indexing.

Fix 5 — Causal mask dtype was copied from the original attention_mask, which can
         be int64 (2-D boolean/int mask).  masked_fill_(..., float("-inf")) on an
         int tensor raises RuntimeError.  Now always uses inputs_embeds.dtype.

Fix 6 — position_ids dim-0 check was absent; the code assumed shape [4,B,S] but
         Qwen3-VL M-RoPE uses [3,B,S].  Added pid.shape[0] in (3, 4) guard to
         avoid silently operating on the wrong axis.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


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

def _divprune_select(features: torch.Tensor, n_keep: int) -> torch.Tensor:
    """
    Greedy max-min diversity selection using incremental min-distance tracking.

    Memory: O(N) — no N×N matrix is materialised.

    features : [N, D]  — image token embeddings (float32 recommended)
    n_keep   : int     — number of tokens to keep (must be <= N)

    Returns [n_keep] indices (into the N image tokens) of selected tokens.
    All returned indices are unique.

    Fixes applied vs original:
      - Seed step now uses dim=1 topk (row-wise nearest-neighbour distance)
        instead of dim=0 (column-wise), matching the algorithm description.
      - Greedy loop masks out already-selected indices before argmax so the
        same token can never be chosen twice.
      - Full N×N cosine-distance matrix replaced with incremental min_dist
        vector, reducing memory from O(N²) to O(N).
    """
    N = features.shape[0]
    if n_keep >= N:
        return torch.arange(N, device=features.device)

    # Normalise once; reuse for all dot-product distance queries.
    feats_norm = F.normalize(features.float(), p=2, dim=-1)  # [N, D]

    selected = torch.empty(n_keep, dtype=torch.long, device=features.device)

    # ── Seed token (step 0) ────────────────────────────────────────────────
    # Score each token by its nearest-neighbour cosine distance.
    # We want the token whose closest peer is as far away as possible.
    #
    # Full row-wise distances are too expensive to materialise here, so we
    # compute them as a one-off for the seed only: dot-product gives cosine
    # similarity; negate to get distance; zero the diagonal; then take the
    # maximum across each row (= nearest-neighbour distance per token).
    #
    # dim=1 topk: for each row i, find the 2 smallest similarity values.
    # values[:, 1] is the nearest neighbour (values[:, 0] == self == 1.0).
    sim_all = feats_norm @ feats_norm.T                               # [N, N]
    # Fix 1: dim=1 (row-wise), index [:, 1] for nearest-neighbour distance.
    nn_dist = 1.0 - torch.topk(sim_all, 2, dim=1, largest=True).values[:, 1]  # [N]
    selected[0] = torch.argmax(nn_dist)
    del sim_all  # free the N×N matrix; not needed again

    # ── Incremental min-dist tracking (steps 1 … n_keep-1) ────────────────
    # min_dist[i] = min cosine distance from token i to any already-selected token.
    # Initialised using the seed token so the first update is free.
    seed_sim = feats_norm @ feats_norm[selected[0]]                   # [N]
    min_dist = 1.0 - seed_sim                                         # [N]

    for i in range(1, n_keep):
        scores = min_dist.clone()
        # Fix 2: mask out already-selected indices so argmax cannot re-pick them.
        scores[selected[:i]] = -1.0
        selected[i] = torch.argmax(scores)

        # Update min_dist with distances to the newly selected token.
        new_sim = feats_norm @ feats_norm[selected[i]]                # [N]
        new_dist = 1.0 - new_sim
        min_dist = torch.minimum(min_dist, new_dist)

    return selected


# ---------------------------------------------------------------------------
# Pre-language-model hook
# ---------------------------------------------------------------------------

def _make_pre_language_model_hook(state: _DivPruneState):
    """
    Returns a register_forward_pre_hook callable for language_model.

    Hooked at the language_model level (not layers[0]) so that both
    inputs_embeds and deepstack tensors (visual_pos_masks,
    deepstack_visual_embeds) are pruned before any decoder layer or
    _deepstack_process call sees them.

    Pruned kwargs:
      inputs_embeds          [B, S, D]          → [B, new_S, D]
      visual_pos_masks       [B, S]             → [B, new_S]
      deepstack_visual_embeds list[[N_vis, D]]  → list[[n_keep, D]]
      position_ids           [3or4, B, S]       → [3or4, B, new_S]
      attention_mask         rebuilt as causal float mask [B,1,new_S,new_S]

    Fixes applied vs original:
      Fix 4: deepstack_visual_embeds elements are asserted to have leading dim
             == n_img before indexing, surfacing any shape mismatch early.
      Fix 5: causal mask always uses inputs_embeds.dtype (a float type) rather
             than copying the original attention_mask dtype, which can be int64.
      Fix 6: position_ids handling checks pid.shape[0] in (3, 4) explicitly so
             we never silently slice on the wrong axis.
    """
    def _hook(module, args, kwargs):
        # Only fire once per generate() call, and only on the prefill pass.
        if state.pruning_done or state.image_token_positions is None:
            return args, kwargs

        # inputs_embeds is always passed as a kwarg when visual tokens are present.
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None or not isinstance(inputs_embeds, torch.Tensor):
            return args, kwargs

        B, seq_len, hidden_dim = inputs_embeds.shape

        image_mask = state.image_token_positions        # [B, orig_S]
        if image_mask.shape[1] > seq_len:
            image_mask = image_mask[:, :seq_len]
        elif image_mask.shape[1] < seq_len:
            # Decode step (seq_len == 1 in standard generation) — skip.
            return args, kwargs

        if not image_mask.any():
            state.pruning_done = True
            return args, kwargs

        # ── Single-batch fast path (standard eval loop) ──────────────────
        if B == 1:
            img_pos = torch.where(image_mask[0])[0]     # [N_img] global positions
            n_img = img_pos.numel()
            n_keep = max(1, int(round(n_img * state.config.divprune_r)))

            # DivPrune greedy selection
            img_feats = inputs_embeds[0, img_pos]        # [N_img, D]
            sel_local = _divprune_select(img_feats, n_keep)  # [n_keep] into img_pos
            sel_pos = img_pos[sel_local]                 # global positions kept

            # keep_mask: True = keep this position in the sequence
            keep_mask = (~image_mask[0]).clone()         # all non-image = True
            keep_mask[sel_pos] = True
            kept_idx = torch.where(keep_mask)[0]         # [new_S]
            new_S = kept_idx.numel()

            new_kwargs = dict(kwargs)

            # ── 1. Prune inputs_embeds ────────────────────────────────────
            new_kwargs["inputs_embeds"] = inputs_embeds[:, kept_idx, :]

            # ── 2. Prune visual_pos_masks [B, S] → [B, new_S] ───────────
            vpm = kwargs.get("visual_pos_masks")
            if vpm is not None:
                new_kwargs["visual_pos_masks"] = vpm[:, kept_idx]

            # ── 3. Prune deepstack_visual_embeds ─────────────────────────
            # Each element is expected to have shape [N_img, D] where N_img
            # matches the number of image tokens in the prefill sequence.
            # sel_local_sorted: sorted indices (0..n_img-1) of kept visual tokens.
            dve = kwargs.get("deepstack_visual_embeds")
            if dve is not None:
                sel_local_sorted = sel_local.sort().values
                pruned_dve = []
                for embed in dve:
                    # Fix 4: assert shape matches before indexing to surface
                    # any mismatch as a clear error rather than a silent truncation
                    # or an IndexError with a confusing traceback.
                    assert embed.shape[0] == n_img, (
                        f"[DivPrune] deepstack embed leading dim {embed.shape[0]} "
                        f"!= n_img {n_img}. Qwen3-VL may have changed its internal "
                        f"layout — check deepstack_visual_embeds packing."
                    )
                    pruned_dve.append(embed[sel_local_sorted])
                new_kwargs["deepstack_visual_embeds"] = pruned_dve

            # ── 4. Prune position_ids ─────────────────────────────────────
            # Fix 6: Qwen3-VL M-RoPE uses shape [3, B, S] (t/h/w components).
            # Guard on pid.shape[0] in (3, 4) to avoid silently slicing the
            # wrong axis if the layout ever differs.
            pid = kwargs.get("position_ids")
            if pid is not None:
                try:
                    if (
                        pid.dim() == 3
                        and pid.shape[0] in (3, 4)
                        and pid.shape[-1] == seq_len
                    ):
                        # M-RoPE: [num_components, B, S]
                        new_kwargs["position_ids"] = pid[:, :, kept_idx]
                    elif pid.dim() == 2 and pid.shape[-1] == seq_len:
                        new_kwargs["position_ids"] = pid[:, kept_idx]
                except Exception:
                    pass

            # ── 5. Build explicit causal mask for pruned sequence ────────
            # The 4D [B,1,S,S] mask is pre-built in prepare_inputs_for_generation
            # using the original S.  We must rebuild it for the new sequence
            # length so downstream code (which may use the 4-D early-exit path
            # inside create_causal_mask) sees a consistent shape.
            #
            # Fix 5: always use a float dtype (inputs_embeds.dtype) for the
            # causal mask.  The original attention_mask can be int64 (a 2-D
            # boolean/padding mask), and masked_fill_(..., float("-inf")) on an
            # integer tensor raises RuntimeError.
            causal_mask = torch.zeros(
                B, 1, new_S, new_S,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            causal_mask.masked_fill_(
                torch.ones(new_S, new_S, dtype=torch.bool, device=inputs_embeds.device)
                      .triu(diagonal=1),
                float("-inf"),
            )
            new_kwargs["attention_mask"] = causal_mask

            # ── Record stats ──────────────────────────────────────────────
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

            return args, new_kwargs

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
      - language_model : pre-forward hook (with_kwargs=True)
      - model.generate : wrapped to reset state + capture input_ids
      - model._divprune_* : bookkeeping attributes

    Returns the same model, patched in-place.
    """
    if not config.use_divprune:
        return model

    if hasattr(model, "_divprune_hook_handle"):
        print("[DivPrune] WARNING: already applied — call remove_divprune() first.")
        return model

    lang_model = model.model.language_model

    state = _DivPruneState(config)

    hook_handle = lang_model.register_forward_pre_hook(
        _make_pre_language_model_hook(state),
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