"""
=============================================================================
  Qwen3-VL 4B  —  FULLY ANNOTATED INFERENCE CODE
  Every layer named, every tensor shaped, KV cache fill/access marked.
=============================================================================

LAYER MAP (top-level):
  Qwen3VLForConditionalGeneration
  ├── visual                          ← Qwen3VLVisionModel   (SigLIP-2 ViT, 300M)
  │   ├── patch_embed                 ← Conv3d patch tokeniser
  │   ├── rotary_pos_emb             ← 2D-RoPE for image patches
  │   └── blocks[0..31]              ← 32 ViT Transformer layers
  │       ├── norm1                   ← LayerNorm
  │       ├── attn                    ← Qwen3VLVisionSdpaAttention
  │       │   ├── qkv                 ← nn.Linear  (fused Q+K+V)
  │       │   └── proj                ← nn.Linear  (output projection)
  │       ├── norm2                   ← LayerNorm
  │       └── mlp                     ← Qwen3VLMLP
  │           ├── fc1                 ← nn.Linear
  │           ├── act                 ← nn.GELU
  │           └── fc2                 ← nn.Linear
  ├── merger                          ← Qwen3VLPatchMerger  (MLP projector)
  │   ├── norm                        ← LayerNorm
  │   ├── linear_fc1                  ← nn.Linear  (4x spatial compress)
  │   ├── act_fn                      ← nn.GELU
  │   └── linear_fc2                  ← nn.Linear  (project to LLM dim)
  ├── deepstack_mergers[0..2]        ← 3 × Qwen3VLPatchMerger (one per ViT level)
  └── model                           ← Qwen3VLTextModel  (LLM backbone, 4B)
      ├── embed_tokens                ← nn.Embedding  [vocab_size, hidden_size]
      └── layers[0..35]              ← 36 × Qwen3VLTextDecoderLayer
          ├── input_layernorm         ← Qwen3VLTextRMSNorm
          ├── self_attn               ← Qwen3VLTextSdpaAttention
          │   ├── q_proj              ← nn.Linear  [hidden → num_heads * head_dim]
          │   ├── k_proj              ← nn.Linear  [hidden → num_kv_heads * head_dim]
          │   ├── v_proj              ← nn.Linear  [hidden → num_kv_heads * head_dim]
          │   ├── q_norm              ← Qwen3VLTextRMSNorm  (per-head QK norm)
          │   ├── k_norm              ← Qwen3VLTextRMSNorm  (per-head QK norm)
          │   ├── o_proj              ← nn.Linear  [num_heads * head_dim → hidden]
          │   └── rotary_emb          ← Qwen3VLInterlevedMRoPE
          ├── post_attention_layernorm← Qwen3VLTextRMSNorm
          └── mlp                     ← Qwen3VLTextMLP  (SwiGLU)
              ├── gate_proj           ← nn.Linear  [hidden → intermediate]
              ├── up_proj             ← nn.Linear  [hidden → intermediate]
              ├── act_fn              ← nn.SiLU
              └── down_proj           ← nn.Linear  [intermediate → hidden]

4B CONFIG (text backbone):
  hidden_size         = 2560
  intermediate_size   = 9728  (SwiGLU FFN inner)
  num_hidden_layers   = 36
  num_attention_heads = 32
  num_kv_heads        = 8     (GQA: 4 query heads share 1 kv head)
  head_dim            = 80    (hidden / num_heads = 2560 / 32)
  max_position_embeddings = 32768
  rope_theta          = 1_000_000
  vocab_size          = 151_936

VISION CONFIG (SigLIP-2 Large):
  hidden_size         = 1152
  intermediate_size   = 4304
  num_hidden_layers   = 27  (ViT-Large depth)
  num_heads           = 16
  patch_size          = 14   (spatial)
  temporal_patch_size = 2    (for video: 2 frames merged into 1 patch)

KV CACHE LEGEND (search for these markers in the code):
  <<KV_FILL>>    — where K and V are computed and written into the cache
  <<KV_READ>>    — where K and V are read back from the cache (past tokens)
  <<KV_UPDATE>>  — the cache.update() call that does both atomically
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
#  0.  DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
# pip install "transformers>=4.57.0" torch torchvision Pillow requests qwen-vl-utils

import torch
import requests
from io import BytesIO
from PIL import Image

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)

# ─────────────────────────────────────────────────────────────────────────────
#  1.  MODEL LOAD
#      We load in bfloat16 to halve VRAM usage (4B needs ~8 GB at bf16).
#      device_map="auto" shards across GPUs if you have multiple.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

print("Loading model weights from HuggingFace Hub…")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",            # automatically places layers on available devices
    # attn_implementation="flash_attention_2",  # uncomment if flash-attn is installed
)
model.eval()   # disables dropout; critical for deterministic inference

processor = AutoProcessor.from_pretrained(MODEL_ID)

# ─────────────────────────────────────────────────────────────────────────────
#  2.  PRINT THE FULL LAYER TREE
#      This is the most direct way to see every named sub-module.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  FULL MODEL ARCHITECTURE  (every layer, every sub-module)")
print("="*72)
print(model)
# Prints the complete nn.Module tree. Equivalent to iterating:
#   for name, module in model.named_modules():
#       print(f"{name}: {type(module).__name__}")

# For a cleaner, size-annotated view:
print("\n" + "="*72)
print("  PARAMETER COUNT PER TOP-LEVEL MODULE")
print("="*72)
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"  {name:<30}  {params/1e6:>8.1f} M params")

# ─────────────────────────────────────────────────────────────────────────────
#  3.  LAYER-BY-LAYER FORWARD PASS WITH HOOKS
#      We register a forward hook on every named sub-module so we can
#      print the input/output tensor shapes as they flow through the model.
#      This is like printf-debugging the entire computation graph.
# ─────────────────────────────────────────────────────────────────────────────

hooks = []          # store handles so we can remove them later
layer_log = []      # accumulate (name, in_shape, out_shape) tuples

def make_hook(name):
    """Returns a forward hook that records tensor shapes for `name`."""
    def hook_fn(module, inputs, output):
        # inputs is a tuple; grab the first tensor input
        def shape_of(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
            if isinstance(x, (list, tuple)):
                return [shape_of(i) for i in x if isinstance(i, torch.Tensor)]
            return type(x).__name__

        in_shape  = shape_of(inputs[0]) if inputs else "—"
        out_shape = shape_of(output[0]) if isinstance(output, tuple) else shape_of(output)
        layer_log.append((name, in_shape, out_shape))
    return hook_fn

# Register hooks on every sub-module
for name, module in model.named_modules():
    if name:  # skip the root module itself
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

# ─────────────────────────────────────────────────────────────────────────────
#  4.  BUILD A SAMPLE INPUT  (image + text question)
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images"
    "/resolve/main/pipeline-cat-chonk.jpeg"
)
print(f"\nDownloading sample image from {IMAGE_URL}…")
response = requests.get(IMAGE_URL, timeout=15)
image = Image.open(BytesIO(response.content)).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": "Describe what you see in this image."},
        ],
    }
]

# apply_chat_template:
#   1. Formats the conversation into the model's expected prompt string
#   2. Tokenises text → input_ids
#   3. Processes image pixels → pixel_values  (resized, normalized, patched)
#   4. Computes image_grid_thw              (temporal, height, width patch counts)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs.pop("token_type_ids", None)   # Qwen3-VL doesn't use token_type_ids
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("\n" + "="*72)
print("  PREPROCESSED INPUT TENSORS")
print("="*72)
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k:<30}  shape={tuple(v.shape)}  dtype={v.dtype}")
    else:
        print(f"  {k:<30}  {v}")

# ─────────────────────────────────────────────────────────────────────────────
#  5.  PREFILL FORWARD PASS  (one shot, no generation yet)
#      This runs the model on the full prompt sequence.
#      The KV cache is FILLED here for all prompt tokens.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  PREFILL PASS  —  running model.forward() on the prompt")
print("  <<KV_FILL>>  happens inside every LLM attention layer here")
print("="*72)

layer_log.clear()   # reset before the forward pass

with torch.no_grad():
    outputs = model(
        **inputs,
        use_cache=True,         # <<KV_FILL>>  instructs every attention block
                                # to compute K, V and store them in past_key_values
        output_hidden_states=True,   # returns hidden states at every layer
        return_dict=True,
    )

# Remove hooks — we only needed them for this one pass
for h in hooks:
    h.remove()

# ─────────────────────────────────────────────────────────────────────────────
#  6.  PRINT LAYER-BY-LAYER TENSOR SHAPES
#      Now we dump the log collected by the hooks.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  LAYER ACTIVATION SHAPES (input_shape → output_shape)")
print("  (Only showing layers with tensor inputs for clarity)")
print("="*72)

# Filter to the most informative layers; skip tiny utility modules
INTERESTING_PREFIXES = (
    "visual.blocks.",
    "visual.patch_embed",
    "visual.rotary_pos_emb",
    "merger.",
    "deepstack_mergers.",
    "model.embed_tokens",
    "model.layers.",
    "lm_head",
)
print(f"  {'LAYER NAME':<52}  {'IN SHAPE':<28}  {'OUT SHAPE'}")
print(f"  {'-'*52}  {'-'*28}  {'-'*28}")
for name, in_shape, out_shape in layer_log:
    if any(name.startswith(p) for p in INTERESTING_PREFIXES):
        if isinstance(in_shape, tuple) and len(in_shape) >= 2:
            print(f"  {name:<52}  {str(in_shape):<28}  {str(out_shape)}")

# ─────────────────────────────────────────────────────────────────────────────
#  7.  INSPECT THE KV CACHE STRUCTURE
#      After the prefill pass, past_key_values holds the cached K and V
#      tensors for every LLM layer. The ViT does NOT use a KV cache
#      (it processes images only once, not autoregressively).
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  KV CACHE ANATOMY  —  outputs.past_key_values")
print("  <<KV_FILL>> was called once per LLM layer during prefill")
print("="*72)

kv_cache = outputs.past_key_values   # transformers DynamicCache object

if kv_cache is not None:
    num_layers = len(kv_cache.key_cache)
    print(f"\n  Cache type        : {type(kv_cache).__name__}")
    print(f"  LLM layers cached : {num_layers}  (one entry per decoder layer)")
    print(f"\n  {'LAYER':<8}  {'KEY shape':<40}  {'VALUE shape':<40}  VRAM (K+V)")
    print(f"  {'-'*8}  {'-'*40}  {'-'*40}  {'-'*12}")

    total_bytes = 0
    for i in range(num_layers):
        k = kv_cache.key_cache[i]    # <<KV_READ>> during generation: this tensor
        v = kv_cache.value_cache[i]  # <<KV_READ>> is passed back to attention
        # Shape: [batch, num_kv_heads, seq_len, head_dim]
        kbytes = k.numel() * k.element_size() + v.numel() * v.element_size()
        total_bytes += kbytes
        print(
            f"  layer {i:<3}  "
            f"K={str(tuple(k.shape)):<38}  "
            f"V={str(tuple(v.shape)):<38}  "
            f"{kbytes/1024:.1f} KB"
        )

    print(f"\n  Total KV cache size after prefill: {total_bytes/1024**2:.2f} MB")
    print(f"\n  KV cache shape breakdown:")
    k0 = kv_cache.key_cache[0]
    print(f"    dim 0 = batch_size         = {k0.shape[0]}")
    print(f"    dim 1 = num_kv_heads       = {k0.shape[1]}  (GQA: 8 kv heads shared by 32 query heads)")
    print(f"    dim 2 = seq_len (prefill)  = {k0.shape[2]}  tokens in prompt")
    print(f"    dim 3 = head_dim           = {k0.shape[3]}")

# ─────────────────────────────────────────────────────────────────────────────
#  8.  WHERE EXACTLY IS KV CACHE FILLED AND ACCESSED?
#      The critical lines in transformers' Qwen3VLTextSdpaAttention.forward():
#
#  PSEUDOCODE (mirrors actual HuggingFace source):
#
#  def forward(self, hidden_states, position_ids, past_key_values, ...):
#
#      # ── PROJECT Q, K, V ──────────────────────────────────────────────────
#      query_states = self.q_proj(hidden_states)     # [B, seq, num_heads*head_dim]
#      key_states   = self.k_proj(hidden_states)     # [B, seq, num_kv_heads*head_dim]
#      value_states = self.v_proj(hidden_states)     # [B, seq, num_kv_heads*head_dim]
#
#      # reshape to [B, heads, seq, head_dim]
#      query_states = query_states.view(B, seq, num_heads, head_dim).transpose(1,2)
#      key_states   = key_states.view(B, seq, num_kv_heads, head_dim).transpose(1,2)
#      value_states = value_states.view(B, seq, num_kv_heads, head_dim).transpose(1,2)
#
#      # ── QK NORM (Qwen3-VL specific) ──────────────────────────────────────
#      # Normalise Q and K per-head before RoPE. Stabilises training.
#      query_states = self.q_norm(query_states)
#      key_states   = self.k_norm(key_states)
#
#      # ── APPLY INTERLEAVED MRoPE ──────────────────────────────────────────
#      cos, sin = self.rotary_emb(value_states, position_ids)
#      query_states, key_states = apply_multimodal_rotary_pos_emb(
#          query_states, key_states, cos, sin, mrope_section
#      )
#      # mrope_section defines the interleaved t/h/w dimension assignment
#
#      # ── <<KV_UPDATE>> ─────────────────────────────────────────────────────
#      # This single call does two things atomically:
#      #   WRITE (fill):  concatenate current K, V onto the cache tensor
#      #   READ  (access): return the full K, V including all past tokens
#      key_states, value_states = past_key_values.update(
#          key_states,   # new K for current tokens  <<KV_FILL>>
#          value_states, # new V for current tokens  <<KV_FILL>>
#          self.layer_idx,  # which cache slot to use
#      )
#      # After this call:
#      #   key_states   = [B, num_kv_heads, past_seq + curr_seq, head_dim]  <<KV_READ>>
#      #   value_states = [B, num_kv_heads, past_seq + curr_seq, head_dim]  <<KV_READ>>
#
#      # ── GQA: EXPAND KV HEADS ─────────────────────────────────────────────
#      # 8 kv heads → 32 query heads (repeat each kv head 4 times)
#      key_states   = repeat_kv(key_states,   num_heads // num_kv_heads)
#      value_states = repeat_kv(value_states, num_heads // num_kv_heads)
#
#      # ── SCALED DOT-PRODUCT ATTENTION ─────────────────────────────────────
#      # Uses torch.nn.functional.scaled_dot_product_attention (SDPA)
#      # which dispatches to Flash Attention if available
#      attn_output = F.scaled_dot_product_attention(
#          query_states,   # [B, num_heads, curr_seq, head_dim]
#          key_states,     # [B, num_heads, full_seq, head_dim]  <<KV_READ>>
#          value_states,   # [B, num_heads, full_seq, head_dim]  <<KV_READ>>
#          attn_mask=causal_mask,
#          scale=1.0 / math.sqrt(head_dim),
#      )
#
#      # ── OUTPUT PROJECTION ─────────────────────────────────────────────────
#      attn_output = attn_output.transpose(1,2).reshape(B, curr_seq, hidden_size)
#      return self.o_proj(attn_output), past_key_values
#
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*72)
print("  ATTENTION FLOW SUMMARY (one LLM decoder layer, prefill vs decode)")
print("="*72)

print("""
  PREFILL (processes all N prompt tokens at once):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  hidden_states [B, N, 2560]                                          │
  │       ↓  q_proj / k_proj / v_proj                                    │
  │  Q [B,32,N,80]   K [B,8,N,80]   V [B,8,N,80]                       │
  │       ↓  q_norm + k_norm  (per-head RMSNorm)                         │
  │       ↓  Interleaved MRoPE (position_ids carry t,h,w for visuals)    │
  │       ↓  <<KV_FILL>>  cache.update(K, V, layer_idx)                  │
  │            writes K,V into past_key_values[layer_idx]                 │
  │       ↓  <<KV_READ>>  returns K [B,8,N,80], V [B,8,N,80]            │
  │       ↓  repeat_kv → K [B,32,N,80], V [B,32,N,80]  (GQA expand)    │
  │       ↓  SDPA(Q, K, V) → attn_out [B,32,N,80]                       │
  │       ↓  o_proj → [B, N, 2560]                                       │
  └──────────────────────────────────────────────────────────────────────┘

  DECODE (generates one new token at a time):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  hidden_states [B, 1, 2560]  ← only the newly generated token        │
  │       ↓  q_proj / k_proj / v_proj                                    │
  │  Q [B,32,1,80]   K_new [B,8,1,80]   V_new [B,8,1,80]               │
  │       ↓  q_norm + k_norm + MRoPE                                     │
  │       ↓  <<KV_FILL>>  cache.update(K_new, V_new, layer_idx)          │
  │            APPENDS to existing cache:                                  │
  │            K_cached: [B,8,N,80] → [B,8,N+1,80]                      │
  │            V_cached: [B,8,N,80] → [B,8,N+1,80]                      │
  │       ↓  <<KV_READ>>  returns full K [B,8,N+1,80], V [B,8,N+1,80]  │
  │       ↓  repeat_kv → [B,32,N+1,80]                                  │
  │       ↓  SDPA(Q[1 token], K[N+1 tokens], V[N+1 tokens])             │
  │            Q attends to ALL previous tokens — O(1) compute per step  │
  │       ↓  o_proj → [B, 1, 2560]                                       │
  └──────────────────────────────────────────────────────────────────────┘

  WITHOUT KV CACHE: every decode step would re-run attention over the
  full growing sequence → O(N²) cost per token.
  WITH KV CACHE:    K and V for all past tokens are pre-computed and
  stored; only the new token's K and V are computed → O(N) total.
""")

# ─────────────────────────────────────────────────────────────────────────────
#  9.  DEEPSTACK KV NOTE
#      DeepStack injects intermediate ViT features into LLM layers 0, 1, 2.
#      These injections happen via residual addition BEFORE the attention
#      of those layers — they do NOT go through the KV cache.
#      The KV cache only exists in the LLM (text) attention layers.
#      The ViT runs ONCE per image and its output is not cached across tokens.
# ─────────────────────────────────────────────────────────────────────────────
print("="*72)
print("  DEEPSTACK & KV CACHE INTERACTION")
print("="*72)
print("""
  DeepStack path (NOT cached):
    ViT layer 4  → deepstack_mergers[0] → ADD to LLM layer 0 hidden_states
    ViT layer 16 → deepstack_mergers[1] → ADD to LLM layer 1 hidden_states
    ViT layer 27 → deepstack_mergers[2] → ADD to LLM layer 2 hidden_states

  These additions happen before attention in layers 0/1/2 of the LLM.
  Because the image is only processed once (at prefill), DeepStack features
  become part of the initial hidden states and flow through the KV cache
  indirectly — they influence what K and V look like for those layers,
  but the DeepStack mergers themselves run zero times during decode.

  So:
    ViT           → runs ONCE at prefill, no KV cache of its own
    Merger        → runs ONCE at prefill, no KV cache of its own
    DeepStack     → runs ONCE at prefill, no KV cache of its own
    LLM attention → KV cache FILLED at prefill, READ+APPENDED at every decode step
""")

# ─────────────────────────────────────────────────────────────────────────────
#  10.  AUTOREGRESSIVE GENERATION  (uses the filled KV cache)
#       This is the standard .generate() loop. Internally it calls
#       model.forward() with use_cache=True on each new token, reading
#       past_key_values <<KV_READ>> and appending to it <<KV_FILL>>.
# ─────────────────────────────────────────────────────────────────────────────
print("="*72)
print("  GENERATION  —  autoregressive decode using filled KV cache")
print("  Each decode step: <<KV_READ>> past tokens + <<KV_FILL>> new token")
print("="*72)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,          # <<KV_READ>> + <<KV_FILL>> at every decode step
        do_sample=False,         # greedy decode (deterministic)
        temperature=None,
        top_p=None,
        # past_key_values=outputs.past_key_values,  # can pass pre-filled cache
        # to skip re-running prefill — useful in multi-turn chat
    )

# Trim the prompt tokens from the output
input_len = inputs["input_ids"].shape[1]
new_token_ids = generated_ids[:, input_len:]

output_text = processor.batch_decode(
    new_token_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(f"\n  Generated {new_token_ids.shape[1]} new tokens")
print(f"\n  MODEL RESPONSE:\n  {'─'*60}")
print(f"  {output_text[0]}")
print(f"  {'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
#  11.  MULTI-TURN CHAT  —  reusing the KV cache across turns
#       For multi-turn use, you can pass the pre-filled past_key_values
#       back into model.generate() on the next turn. This avoids re-encoding
#       the entire history — only the new user message is prefilled.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  MULTI-TURN CACHE REUSE PATTERN  (production pattern)")
print("="*72)
print("""
  # Turn 1 — prefill image + question, get cache
  out1 = model.generate(**turn1_inputs, use_cache=True,
                         return_dict_in_generate=True)
  cache = out1.past_key_values   # DynamicCache, fully populated

  # Turn 2 — pass the cache so we skip re-encoding turn 1
  # Only the new user message is prefilled; all previous K/V already stored
  out2 = model.generate(**turn2_inputs, use_cache=True,
                         past_key_values=cache,          # <<KV_READ>> from turn 1
                         return_dict_in_generate=True)
  cache = out2.past_key_values   # cache now includes turn 2 as well

  # Each turn: O(new_tokens_only) prefill instead of O(full_history)
""")

# ─────────────────────────────────────────────────────────────────────────────
#  12.  QUICK REFERENCE: find KV cache code in HuggingFace source
# ─────────────────────────────────────────────────────────────────────────────
print("="*72)
print("  SOURCE CODE POINTERS  (HuggingFace transformers)")
print("="*72)
print("""
  File:  transformers/models/qwen3_vl/modeling_qwen3_vl.py

  class Qwen3VLTextSdpaAttention(nn.Module):
      def forward(self, hidden_states, position_ids, past_key_values, ...):

          # Lines ~650-680 (approx):
          key_states, value_states = past_key_values.update(   # <<KV_UPDATE>>
              key_states, value_states, self.layer_idx
          )
          # .update() is defined in:
          #   transformers/cache_utils.py  class DynamicCache
          #   def update(self, key_states, value_states, layer_idx):
          #       self.key_cache[layer_idx]   = cat([existing, key_states],   dim=2)
          #       self.value_cache[layer_idx] = cat([existing, value_states], dim=2)
          #       return self.key_cache[layer_idx], self.value_cache[layer_idx]

  File:  transformers/cache_utils.py
      class DynamicCache:
          key_cache   : List[Tensor]   # [num_layers][B, kv_heads, seq, head_dim]
          value_cache : List[Tensor]   # same shape
          def update(self, key, value, layer_idx) -> (Tensor, Tensor):
              ...
""")

print("\n✓ Done. All layers printed above; KV cache anatomy fully shown.")