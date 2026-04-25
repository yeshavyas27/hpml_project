# =============================================
# Install dependencies first:
# pip install git+https://github.com/huggingface/transformers
# pip install torch torchvision accelerate pillow requests
# pip install flash-attn --no-build-isolation  # optional but recommended
# =============================================

import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ── 1. Load Model & Processor ──────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Option A: Standard load (auto dtype, auto device map)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",      # uses bfloat16 on CUDA automatically
    device_map="auto",       # spreads across all available GPUs
)

# Option B: Explicit bfloat16 + FlashAttention-2 (faster, less VRAM)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained(MODEL_ID)

print(f"Model loaded on: {next(model.parameters()).device}")
print(f"Dtype: {next(model.parameters()).dtype}")


# ── 2. Helper: Load image from URL or local path ───────────────────────────────

def load_image(source: str) -> Image.Image:
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(source).convert("RGB")


# ── 3. Inference Function ──────────────────────────────────────────────────────

def run_inference(
    image_source: str,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> str:
    """
    Run vision-language inference.

    Args:
        image_source: URL or local file path to an image.
        prompt:       Text question / instruction about the image.
        max_new_tokens: Max tokens to generate.
        temperature, top_p, top_k: Sampling parameters.

    Returns:
        Generated text string.
    """

    # Build the chat message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_source},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    # Tokenize with the processor's chat template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0,
        )

    # Strip the input tokens from the output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], generated_ids)
    ]

    output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output[0]


# ── 4. Text-only Inference (no image) ─────────────────────────────────────────

def run_text_inference(prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ── 5. Examples ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # --- Example 1: Describe an image from a URL ---
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    response = run_inference(image_url, "Describe this image in detail.")
    print("=== Image Description ===")
    print(response)

    # --- Example 2: Ask a specific question about an image ---
    response = run_inference(
        image_url,
        "What objects can you identify in this image? List them.",
        max_new_tokens=256,
    )
    print("\n=== Object Identification ===")
    print(response)

    # --- Example 3: OCR / text extraction ---
    response = run_inference(
        image_url,
        "Extract and transcribe any visible text in the image.",
        max_new_tokens=256,
    )
    print("\n=== OCR Output ===")
    print(response)

    # --- Example 4: Local image file ---
    # response = run_inference("/path/to/your/image.jpg", "What is in this image?")
    # print(response)

    # --- Example 5: Text-only query ---
    response = run_text_inference("Explain the difference between CNN and Vision Transformer.")
    print("\n=== Text-only Query ===")
    print(response)

    # --- Example 6: Multi-turn conversation with an image ---
    print("\n=== Multi-turn Conversation ===")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "What is happening in this scene?"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    first_response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    print("Turn 1:", first_response)

    # Append assistant response and ask a follow-up
    messages.append({"role": "assistant", "content": first_response})
    messages.append({"role": "user",      "content": "Can you be more specific about the background?"})

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    second_response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    print("Turn 2:", second_response)