import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"


def load_model_and_processor():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor