from __future__ import annotations

import gc
import json
import platform
import resource
import time
from pathlib import Path

import torch
import torchvision
import transformers
from transformers import (
    AutoProcessor,
    HunYuanVLForConditionalGeneration,
)


MODEL_DIR = (
    Path.home()
    / "work/hunyuanocr/models/HunyuanOCR-1.5"
)

PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)


def peak_rss_gib() -> float:
    """Linux ru_maxrss is reported in KiB."""
    rss_kib = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    return rss_kib / 1024.0 / 1024.0


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"模型目录不存在：{MODEL_DIR}"
        )

    model_file = MODEL_DIR / "model.safetensors"

    if not model_file.is_file():
        raise FileNotFoundError(
            f"权重文件不存在：{model_file}"
        )

    torch.set_grad_enabled(False)

    # 先使用9线程，避免20线程全部占满系统。
    torch.set_num_threads(9)

    print("===== Environment =====", flush=True)
    print("Python      :", platform.python_version(), flush=True)
    print("PyTorch     :", torch.__version__, flush=True)
    print("torchvision :", torchvision.__version__, flush=True)
    print("transformers:", transformers.__version__, flush=True)
    print("CUDA        :", torch.cuda.is_available(), flush=True)
    print("Threads     :", torch.get_num_threads(), flush=True)
    print("Model dir   :", MODEL_DIR, flush=True)
    print(
        "Initial peak RSS:",
        f"{peak_rss_gib():.3f} GiB",
        flush=True,
    )

    print("\n===== Load processor =====", flush=True)

    processor_start = time.perf_counter()

    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        use_fast=False,
        local_files_only=True,
    )

    processor_seconds = (
        time.perf_counter() - processor_start
    )

    print(
        "Processor class:",
        processor.__class__.__name__,
        flush=True,
    )
    print(
        "Processor load time:",
        f"{processor_seconds:.3f} s",
        flush=True,
    )
    print(
        "Peak RSS after processor:",
        f"{peak_rss_gib():.3f} GiB",
        flush=True,
    )

    print("\n===== Load model =====", flush=True)
    print(
        "Loading as CPU FP32 + eager attention...",
        flush=True,
    )

    model_start = time.perf_counter()

    model = (
        HunYuanVLForConditionalGeneration
        .from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    )

    model.eval()

    model_seconds = (
        time.perf_counter() - model_start
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    first_parameter = next(model.parameters())

    print("Model loaded successfully.", flush=True)
    print(
        "Model class:",
        model.__class__.__name__,
        flush=True,
    )
    print(
        "Config model_type:",
        model.config.model_type,
        flush=True,
    )
    print(
        "Parameter dtype:",
        first_parameter.dtype,
        flush=True,
    )
    print(
        "Parameter device:",
        first_parameter.device,
        flush=True,
    )
    print(
        "Parameter count:",
        f"{parameter_count:,}",
        flush=True,
    )
    print(
        "Trainable count:",
        f"{trainable_count:,}",
        flush=True,
    )
    print(
        "Model load time:",
        f"{model_seconds:.3f} s",
        flush=True,
    )
    print(
        "Peak RSS after model:",
        f"{peak_rss_gib():.3f} GiB",
        flush=True,
    )

    report = {
        "model_dir": str(MODEL_DIR),
        "model_class": model.__class__.__name__,
        "processor_class": processor.__class__.__name__,
        "model_type": model.config.model_type,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "parameter_dtype": str(first_parameter.dtype),
        "parameter_device": str(first_parameter.device),
        "processor_load_seconds": processor_seconds,
        "model_load_seconds": model_seconds,
        "peak_rss_gib": peak_rss_gib(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }

    report_path = (
        PROJECT_DIR
        / "docs/model_load_cpu_fp32.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "\nReport saved to:",
        report_path,
        flush=True,
    )
    print(
        "✅ HunyuanOCR CPU FP32 模型加载成功。",
        flush=True,
    )

    del model
    del processor
    gc.collect()


if __name__ == "__main__":
    main()
