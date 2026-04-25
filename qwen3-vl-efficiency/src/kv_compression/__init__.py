"""
KV cache compression methods for Qwen3-VL.

Four methods are exposed, all training-free and implemented via the `kvpress`
library of NVIDIA:

    h2o        - ExpectedAttentionPress (heavy-hitter style)
    streaming  - StreamingLLMPress       (attention sinks + recent window)
    snapkv     - SnapKVPress              (prompt-aware via last-window attention)
    pyramid    - PyramidKVPress           (SnapKV with per-layer pyramid budget)
    modality   - ModalityAwarePress       (novel: separate image / text budgets)

Factory helpers:

    make_h2o_press(compression_ratio)
    make_streaming_press(compression_ratio, n_sink)
    make_snapkv_press(compression_ratio, window_size, kernel_size)
    make_pyramid_press(compression_ratio, window_size, kernel_size, beta)
    make_modality_aware_press(image_compression_ratio, text_compression_ratio, ...)
    make_press(method, **kwargs)   # dispatcher

All presses are context managers, used identically inside the eval loop:

    press = make_press("h2o", compression_ratio=0.5)
    with press(model):
        output = model.generate(**inputs, max_new_tokens=64)
"""
from .h2o import make_h2o_press
from .streaming import make_streaming_press
from .snapkv import make_snapkv_press, make_pyramid_press
from .modality import make_modality_aware_press, ModalityAwarePress

__all__ = [
    "make_h2o_press",
    "make_streaming_press",
    "make_snapkv_press",
    "make_pyramid_press",
    "make_modality_aware_press",
    "make_press",
    "ModalityAwarePress",
]


def make_press(method: str, **kwargs):
    """
    Dispatcher that returns a kvpress press object given a method name.

    Parameters
    ----------
    method : str
        One of: "h2o", "streaming" / "streamingllm", "snapkv", "pyramid",
        "modality" / "modality_aware".
    **kwargs
        Forwarded to the corresponding factory.  Unknown kwargs are silently
        ignored so that callers can pass a common config blob.
    """
    method = method.lower().replace("-", "_")

    if method == "h2o":
        return make_h2o_press(
            compression_ratio=kwargs.get("compression_ratio", 0.5),
        )

    if method in ("streaming", "streamingllm", "streaming_llm"):
        return make_streaming_press(
            compression_ratio=kwargs.get("compression_ratio", 0.5),
            n_sink=kwargs.get("n_sink", 4),
        )

    if method == "snapkv":
        return make_snapkv_press(
            compression_ratio=kwargs.get("compression_ratio", 0.5),
            window_size=kwargs.get("window_size", 64),
            kernel_size=kwargs.get("kernel_size", 5),
        )

    if method in ("pyramid", "pyramidkv"):
        return make_pyramid_press(
            compression_ratio=kwargs.get("compression_ratio", 0.5),
            window_size=kwargs.get("window_size", 64),
            kernel_size=kwargs.get("kernel_size", 5),
            beta=kwargs.get("beta", 20),
        )

    if method in ("modality", "modality_aware", "modalityaware"):
        return make_modality_aware_press(
            image_compression_ratio=kwargs.get("image_compression_ratio", 0.7),
            text_compression_ratio=kwargs.get("text_compression_ratio", 0.2),
            inner=kwargs.get("inner", "h2o"),
            window_size=kwargs.get("window_size", 64),
        )

    raise ValueError(
        f"Unknown method '{method}'. Choose from: h2o, streaming, snapkv, "
        f"pyramid, modality."
    )
