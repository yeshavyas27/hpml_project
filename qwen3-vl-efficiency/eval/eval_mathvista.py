import os
import json
from datasets import load_dataset
from src.load_model import load_model_and_processor
from src.utils import reset_gpu_memory, get_peak_gpu_memory_mb, timed_inference


RESULTS_PATH = "results/baseline/mathvista_results.jsonl"
DATASET_NAME = "AI4Math/MathVista"
MAX_SAMPLES = 2   # keep small for CPU testing


def generate_answer(model, processor, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=32)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def main():
    os.makedirs("results/baseline", exist_ok=True)

    print("Loading model...")
    model, processor = load_model_and_processor()

    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="test")

    print(f"Running first {MAX_SAMPLES} samples...")

    with open(RESULTS_PATH, "w") as f:
        for idx in range(min(MAX_SAMPLES, len(dataset))):
            sample = dataset[idx]

            # 🔥 IMPORTANT: check fields (may vary)
            question = sample["query"]
            answer = sample.get("answer", "N/A")
            image = sample.get("decoded_image", None)

            reset_gpu_memory()
            prediction, latency_sec = timed_inference(
                generate_answer, model, processor, image, question
            )
            peak_gpu_mem_mb = get_peak_gpu_memory_mb()

            row = {
                "sample_id": idx,
                "dataset": "mathvista",
                "question": question,
                "ground_truth": answer,
                "prediction": prediction,
                "latency_sec": latency_sec,
                "peak_gpu_mem_mb": peak_gpu_mem_mb,
            }

            f.write(json.dumps(row) + "\n")

            print("=" * 80)
            print(f"Sample: {idx}")
            print(f"Question: {question}")
            print(f"Ground truth: {answer}")
            print(f"Prediction: {prediction}")
            print(f"Latency (s): {latency_sec:.3f}")
            print(f"Peak GPU memory (MB): {peak_gpu_mem_mb:.2f}")

    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()