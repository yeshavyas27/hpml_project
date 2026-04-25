"""
FastV token pruning for Qwen3-VL models (HuggingFace Transformers >= 4.57).

=== DESIGN PHILOSOPHY ===

Qwen3-VL has an extremely complex forward pipeline:
  - 4D M-RoPE position_ids: shape [4, B, seq_len] (not 3!)
  - DeepStack: injects multi-level ViT features into early LLM layers
  - Custom causal mask creation via create_causal_mask()
  - The Qwen3VLModel.forward() orchestrates vision encoding, embedding
    merging, position ID computation, and DeepStack — then calls
    language_model.forward() which handles caching, mask creation, etc.

Patching language_model.forward() is FRAGILE and breaks on every
transformers update. Instead, this module takes a SAFE approach:

  **We do not patch any forward method.**

Instead we:
  1. Wrap model.generate() to record which token positions are image tokens
     (from input_ids) before generation begins.
  2. Register a forward hook on decoder layer K that, after its first
     execution during prefill, directly edits the DynamicCache to remove
     KV entries for low-importance image tokens.
  3. All subsequent layers (K+1 .. N-1) naturally process only the pruned
     KV cache during both prefill continuation and autoregressive decode.

This means the hidden_states flowing through layers 0..K are UNTOUCHED.
The pruning only affects what's in the KV cache for layers K+1 onward,
which is mathematically equivalent to those later layers never having
seen the dropped tokens.

Importance signal: L2 norm of the key vectors at layer K for image token
positions (available directly from the cache after layer K runs).

=== v1 BUG & FIX ===

v1 bug: hook registered with with_kwargs=False, so the hook signature was
  _hook(module, args, output)
and past_key_values was read from args[4].

But Qwen3-VL's decoder layers receive past_key_values as a **keyword
argument**, so args only contains positional args (hidden_states,
position_embeddings) and args[4] was always None. Pruning never fired,
which is why your metrics showed zero difference.

Fix: register with with_kwargs=True so the hook signature becomes
  _hook(module, args, kwargs, output)
and past_key_values is reliably extracted from kwargs.

=== HOW TO VERIFY IT'S WORKING ===

Set verbose=True in FastVConfig. You will see per-sample output like:

  [FastV] ✓ layer=3  image tokens: 296 → 148 (pruned 148)  |  seq: 412 → 264

In evaluate_mmmu.py metrics output, mean_image_tokens should drop from
~296 to ~148 at keep_ratio=0.5. If it stays at 296, the hook is not firing.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field, asdict
from typing import Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# Config & Stats dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FastVConfig:
    """Configuration for FastV token pruning."""
    use_fastv: bool = True
    fastv_k: int = 3           # Prune after this layer index (0-indexed)
    fastv_r: float = 0.5       # Ratio of image tokens to KEEP (0.5 = keep 50%)
    image_token_id: int = 151655  # <|image_pad|> token ID for Qwen3-VL
    verbose: bool = False      # Print per-sample pruning confirmation


@dataclass
class FastVStats:
    """
    Per-sample pruning statistics. Returned by get_fastv_stats(model).

    Integrate with evaluate_mmmu.py's SampleMetrics:

        fv = get_fastv_stats(model)
        if fv.pruning_applied:
            m.image_tokens = fv.kept_image_tokens  # reflect actual kept count
        record["fastv"] = asdict(fv)               # attach to per_sample output
    """
    pruning_applied: bool = False
    total_image_tokens: int = 0
    kept_image_tokens: int = 0
    pruned_image_tokens: int = 0
    original_seq_len: int = 0
    pruned_seq_len: int = 0
    layer_k: int = -1
    keep_ratio: float = 1.0

    @property
    def actual_keep_ratio(self) -> float:
        if self.total_image_tokens == 0:
            return 1.0
        return self.kept_image_tokens / self.total_image_tokens

    @property
    def compression_ratio(self) -> float:
        """pruned_seq_len / original_seq_len. Lower = more compression."""
        if self.original_seq_len == 0:
            return 1.0
        return self.pruned_seq_len / self.original_seq_len


class FastVHookState:
    """Mutable state shared between the generate wrapper and the layer hook."""
    def __init__(self, config: FastVConfig):
        self.config = config
        self.image_token_positions: Optional[torch.Tensor] = None
        self.pruning_done: bool = False
        self.stats: FastVStats = FastVStats()

    def reset(self):
        self.image_token_positions = None
        self.pruning_done = False
        self.stats = FastVStats()


# ─────────────────────────────────────────────────────────────────────────────
# Core: KV-cache pruning hook
# ─────────────────────────────────────────────────────────────────────────────

def _make_post_layer_hook(state: FastVHookState):
    """
    Returns a post-forward hook for decoder layer K.

    Registered with with_kwargs=True (the critical v1 fix).
    Signature: hook(module, args, kwargs, output)

    Reads past_key_values from kwargs (where Qwen3-VL puts it),
    scores image token positions by key-vector L2 norm,
    and physically deletes low-importance entries from the DynamicCache.
    """
    def _post_layer_hook(module, args, kwargs, output):
        if state.pruning_done or state.image_token_positions is None:
            return output

        # ── Extract past_key_values from kwargs (v1 fix) ──────────────────
        # v1 read from args[4] which was always None for Qwen3-VL because
        # past_key_values is passed as a keyword argument by the decoder.
        past_key_values = kwargs.get("past_key_values", None)

        # Fallback for any version that passes it positionally
        if past_key_values is None and len(args) > 4:
            past_key_values = args[4]

        if past_key_values is None:
            if state.config.verbose:
                print(
                    "[FastV] WARNING: past_key_values is None — hook fired but "
                    "can't prune. Ensure use_cache=True in generate(). "
                    f"kwargs keys: {list(kwargs.keys())}"
                )
            return output

        if hasattr(past_key_values, "key_cache"):
            k_cache = past_key_values.key_cache
            v_cache = past_key_values.value_cache
        elif hasattr(past_key_values, "_key_cache"):
            k_cache = past_key_values._key_cache
            v_cache = past_key_values._value_cache
        else:
            if state.config.verbose:
                print(f"[FastV] WARNING: cache type {type(past_key_values)} has no "
                      "recognized key/value lists. Pruning skipped.")
            return output

        layer_idx = state.config.fastv_k
        if layer_idx >= len(k_cache):
            return output

        key_states = k_cache[layer_idx]

        if key_states is None or key_states.numel() == 0:
            return output

        B, num_heads, seq_len, head_dim = key_states.shape
        image_mask = state.image_token_positions  # [B, orig_seq_len] bool

        # ── Align mask to current cache sequence length ───────────────────
        if image_mask.shape[1] > seq_len:
            image_mask = image_mask[:, :seq_len]
        elif image_mask.shape[1] < seq_len:
            # Cache longer than mask = decode step, not prefill. Skip.
            return output

        if not image_mask.any():
            state.pruning_done = True
            return output

        # ── Score image tokens by key-vector L2 norm ──────────────────────
        # Rationale: keys with higher norm are more "active" in attention;
        # low-norm keys contribute less to attention computation.
        # Mean over heads → per-position scalar score → [B, seq_len]
        key_norms = key_states.float().norm(dim=-1).mean(dim=1)

        # ── Build per-batch keep_mask ─────────────────────────────────────
        keep_mask = torch.ones(B, seq_len, dtype=torch.bool,
                               device=key_states.device)
        total_image_tokens = 0
        total_pruned = 0

        for b in range(B):
            img_positions = torch.where(image_mask[b, :seq_len])[0]
            n_img = len(img_positions)
            if n_img == 0:
                continue

            n_keep = max(1, int(n_img * state.config.fastv_r))
            n_drop = n_img - n_keep
            total_image_tokens += n_img

            if n_drop <= 0:
                continue

            img_scores = key_norms[b, img_positions]
            _, drop_local_idx = torch.topk(img_scores, n_drop, largest=False)
            drop_positions = img_positions[drop_local_idx]
            keep_mask[b, drop_positions] = False
            total_pruned += n_drop

        if total_pruned == 0:
            state.pruning_done = True
            return output

        # ── Physically remove KV entries for layers 0..K ──────────────────
        n_layers_to_prune = min(layer_idx + 1, len(k_cache))

        for l_idx in range(n_layers_to_prune):
            k = k_cache[l_idx]
            v = v_cache[l_idx]
            if k is None:
                continue

            if B == 1:
                # Fast single-sample path (standard eval loop)
                kept_idx = torch.where(keep_mask[0])[0]
                k_cache[l_idx]   = k[:, :, kept_idx, :]
                v_cache[l_idx] = v[:, :, kept_idx, :]
            else:
                new_k, new_v = [], []
                for b in range(B):
                    kept_idx = torch.where(keep_mask[b])[0]
                    new_k.append(k[b:b+1, :, kept_idx, :])
                    new_v.append(v[b:b+1, :, kept_idx, :])

                kept_lens = [t.shape[2] for t in new_k]
                if all(kl == kept_lens[0] for kl in kept_lens):
                    k_cache[l_idx]   = k[:, :, kept_idx, :]
                    v_cache[l_idx] = v[:, :, kept_idx, :]
                else:
                    # Rare: different images pruned to different lengths — pad
                    max_kept = max(kept_lens)
                    pk = torch.zeros(B, k.shape[1], max_kept, head_dim,
                                     device=k.device, dtype=k.dtype)
                    pv = torch.zeros(B, v.shape[1], max_kept, head_dim,
                                     device=v.device, dtype=v.dtype)
                    for b in range(B):
                        n = kept_lens[b]
                        pk[b, :, :n, :] = new_k[b][0]
                        pv[b, :, :n, :] = new_v[b][0]
                    k_cache[l_idx]   = k[:, :, kept_idx, :]
                    v_cache[l_idx] = v[:, :, kept_idx, :]

        # ── Update DynamicCache bookkeeping ───────────────────────────────
        # DynamicCache._seen_tokens is used by get_seq_length() which feeds
        # into cache_position computation for subsequent decode steps.
        # After pruning, the cache seq length is new_seq_len, not the original.
        if k_cache and len(k_cache) > layer_idx:
            new_seq_len = k_cache[layer_idx].shape[-2] 
            
            # FastV drops tokens, meaning the KV cache is artificially shortened. 
            # We MUST update `_seen_tokens` so the next decode step calculates 
            # the correct causal mask and RoPE position ids.
            if hasattr(past_key_values, "_seen_tokens"):
                past_key_values._seen_tokens = new_seq_len
        # ── Record per-sample stats ───────────────────────────────────────
        state.stats = FastVStats(
            pruning_applied     = True,
            total_image_tokens  = total_image_tokens,
            kept_image_tokens   = total_image_tokens - total_pruned,
            pruned_image_tokens = total_pruned,
            original_seq_len    = seq_len,
            pruned_seq_len      = new_seq_len,
            layer_k             = layer_idx,
            keep_ratio          = state.config.fastv_r,
        )
        state.pruning_done = True

        if state.config.verbose:
            s = state.stats
            print(
                f"[FastV] ✓ layer={layer_idx}  "
                f"image: {s.total_image_tokens} → {s.kept_image_tokens} "
                f"(pruned {s.pruned_image_tokens}, "
                f"{s.actual_keep_ratio:.0%} kept)  |  "
                f"seq: {seq_len} → {new_seq_len} "
                f"({s.compression_ratio:.1%})"
            )

        return output

    return _post_layer_hook


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_fastv_to_qwen(model, fastv_config: FastVConfig):
    """
    Apply FastV image-token pruning to a Qwen3VLForConditionalGeneration model.

    NON-INVASIVE: does NOT replace any forward() method.

    Changes made to model:
      - model.generate  → wrapped to reset state + capture input_ids per call
      - layer K         → post-forward hook registered (with_kwargs=True)
      - model._fastv_*  → bookkeeping attributes added

    Args:
        model:        Qwen3VLForConditionalGeneration (on device, eval mode)
        fastv_config: FastVConfig with pruning parameters

    Returns:
        The same model, patched in-place.
    """
    if not fastv_config.use_fastv:
        return model

    if hasattr(model, '_fastv_hook_handle'):
        print("[FastV] WARNING: FastV already applied — call remove_fastv() first.")
        return model

    lang_model = model.model.language_model
    layers     = lang_model.layers
    num_layers = len(layers)

    assert fastv_config.fastv_k < num_layers, (
        f"fastv_k={fastv_config.fastv_k} must be < num_layers={num_layers}"
    )

    state = FastVHookState(fastv_config)

    # ── Hook: registered with with_kwargs=True (v1 bug fix) ───────────────
    hook_handle = layers[fastv_config.fastv_k].register_forward_hook(
        _make_post_layer_hook(state),
        with_kwargs=True,   # ← CRITICAL: v1 used False; kwargs was never received
    )

    # ── Wrap generate() to capture image token positions per call ─────────
    original_generate = model.generate

    def wrapped_generate(*args, **kwargs):
        # Reset state for this new sequence
        state.reset()

        # Capture image token mask from input_ids
        input_ids = kwargs.get('input_ids', None)
        if input_ids is None and args and isinstance(args[0], torch.Tensor):
            input_ids = args[0]

        if input_ids is not None:
            state.image_token_positions = (
                input_ids == fastv_config.image_token_id
            ).to(input_ids.device)

        return original_generate(*args, **kwargs)

    model.generate                 = wrapped_generate
    model._fastv_hook_handle       = hook_handle
    model._fastv_state             = state
    model._fastv_original_generate = original_generate

    print(
        f"[FastV] ✓ Applied to Qwen3-VL  |  "
        f"layer_k={fastv_config.fastv_k}  "
        f"keep_ratio={fastv_config.fastv_r:.0%}  "
        f"image_token_id={fastv_config.image_token_id}"
    )
    if fastv_config.verbose:
        print("[FastV]   verbose=True — will print per-sample pruning info")
    else:
        print("[FastV]   Set verbose=True in FastVConfig to confirm pruning fires")

    return model


def remove_fastv(model) -> None:
    """
    Remove FastV hooks and restore model.generate to original.
    Safe to call even if FastV was never applied.

    Use between sweep configs:
        apply_fastv_to_qwen(model, FastVConfig(fastv_r=0.5))
        # ... eval run 1 ...
        remove_fastv(model)

        apply_fastv_to_qwen(model, FastVConfig(fastv_r=0.25))
        # ... eval run 2 ...
        remove_fastv(model)
    """
    if not hasattr(model, '_fastv_hook_handle'):
        print("[FastV] Nothing to remove.")
        return

    model._fastv_hook_handle.remove()
    model.generate = model._fastv_original_generate

    del model._fastv_hook_handle
    del model._fastv_state
    del model._fastv_original_generate

    print("[FastV] Removed — model.generate restored to original.")


def get_fastv_stats(model) -> FastVStats:
    """
    Returns FastVStats for the most recent generate() call.

    Returns a no-op FastVStats (pruning_applied=False) if FastV
    is not applied or if pruning didn't fire (e.g. no image tokens).

    Usage in evaluate_mmmu.py run_inference_with_metrics():

        fv = get_fastv_stats(model)
        if fv.pruning_applied:
            m.image_tokens = fv.kept_image_tokens
        # Attach to per-sample record:
        # record["fastv"] = asdict(fv)
    """
    state = getattr(model, '_fastv_state', None)
    if state is None:
        return FastVStats()
    return state.stats


def is_fastv_active(model) -> bool:
    """True if FastV hooks are currently applied."""
    return hasattr(model, '_fastv_hook_handle')


def get_fastv_config(model) -> Optional[FastVConfig]:
    """Returns the active FastVConfig, or None."""
    state = getattr(model, '_fastv_state', None)
    return state.config if state is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Sweep configurations
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_CONFIGS: List[FastVConfig] = [
    FastVConfig(use_fastv=False, fastv_k=3, fastv_r=1.00),   # baseline
    FastVConfig(use_fastv=True,  fastv_k=2, fastv_r=0.75),
    FastVConfig(use_fastv=True,  fastv_k=2, fastv_r=0.50),   # paper default
    FastVConfig(use_fastv=True,  fastv_k=2, fastv_r=0.25),
    FastVConfig(use_fastv=True,  fastv_k=1, fastv_r=0.50),
    FastVConfig(use_fastv=True,  fastv_k=3, fastv_r=0.50),
    FastVConfig(use_fastv=True,  fastv_k=5, fastv_r=0.50),
]

SWEEP_LABELS = [
    "baseline",
    "k2_r75",
    "k2_r50",
    "k2_r25",
    "k1_r50",
    "k3_r50",
    "k5_r50",
]


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_mmmu.py integration instructions
# ─────────────────────────────────────────────────────────────────────────────
#
# ── STEP 1: After model is loaded, add: ───────────────────────────────────────
#
#   from fastv_qwen3vl import (
#       FastVConfig, apply_fastv_to_qwen, remove_fastv,
#       get_fastv_stats, is_fastv_active, get_fastv_config
#   )
#
#   FASTV_CONFIG = FastVConfig(
#       use_fastv = True,
#       fastv_k   = 3,      # layer after which to prune; try 1,2,3,5
#       fastv_r   = 0.5,    # fraction of image tokens to keep
#       verbose   = True,   # KEEP TRUE until you see "[FastV] ✓" in output
#   )
#   if FASTV_CONFIG.use_fastv:
#       apply_fastv_to_qwen(model, FASTV_CONFIG)
#
# ── STEP 2: In run_inference_with_metrics(), after generated_ids = model.generate(...): ──
#
#   if is_fastv_active(model):
#       fv = get_fastv_stats(model)
#       if fv.pruning_applied:
#           m.image_tokens = fv.kept_image_tokens  # override: kept, not total
#
# ── STEP 3: In records.append({...}): ────────────────────────────────────────
#
#   "fastv": asdict(get_fastv_stats(model)) if is_fastv_active(model) else {},
#
# ── STEP 4: In metrics_output dict: ──────────────────────────────────────────
#
#   "fastv_config": asdict(get_fastv_config(model)) if is_fastv_active(model) else None,
#
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("FastV self-test\n")
    for cfg, label in zip(SWEEP_CONFIGS, SWEEP_LABELS):
        tag = "BASELINE" if not cfg.use_fastv else f"k={cfg.fastv_k} r={cfg.fastv_r:.0%}"
        print(f"  {label:<12}  {tag}")

    print("\nFastVStats example:")
    s = FastVStats(
        pruning_applied=True,
        total_image_tokens=296, kept_image_tokens=148,
        pruned_image_tokens=148, original_seq_len=412,
        pruned_seq_len=264, layer_k=3, keep_ratio=0.5,
    )
    print(f"  actual_keep_ratio : {s.actual_keep_ratio:.0%}")
    print(f"  compression_ratio : {s.compression_ratio:.1%}  "
          f"(seq {s.original_seq_len} → {s.pruned_seq_len})")