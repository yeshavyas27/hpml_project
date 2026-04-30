import time
import torch


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def timed_inference(fn, *args, **kwargs):
    start = time.time()
    output = fn(*args, **kwargs)
    end = time.time()
    return output, end - start