from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import HunYuanVLForConditionalGeneration
from transformers.utils import logging


PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

MODEL_DIR = (
    Path.home()
    / "work/hunyuanocr/models/HunyuanOCR-1.5"
)

LAYER23_REFERENCE = (
    PROJECT_DIR
    / "artifacts/decoder_layer23_decode_step2/reference"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "artifacts/decode_tail_step2/reference"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/decode_step2_tail_reference.json"
)

HIDDEN_SIZE = 1024
VOCAB_SIZE = 120818
EXPECTED_TOKEN = 206
TORCH_THREADS = 9


def resolve_module(
    root: Any,
    candidates: tuple[str, ...],
) -> tuple[Any, str]:
    errors: list[str] = []

    for path in candidates:
        current = root

        try:
            for name in path.split("."):
                current = getattr(
                    current,
                    name,
                )

            return current, path
        except AttributeError as error:
            errors.append(
                f"{path}: {error}"
            )

    raise RuntimeError(
        "无法解析模型模块：\n"
        + "\n".join(errors)
    )


def maximum_abs_error(
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    if actual.shape != expected.shape:
        raise RuntimeError(
            "数组形状不一致："
            f"{actual.shape} != {expected.shape}"
        )

    return float(
        np.max(
            np.abs(
                actual.astype(np.float64)
                - expected.astype(np.float64)
            )
        )
    )


def main() -> None:
    logging.disable_progress_bar()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(
        TORCH_THREADS
    )

    hidden_path = (
        LAYER23_REFERENCE
        / "layer23_output_f32.bin"
    )

    expected_logits_path = (
        LAYER23_REFERENCE
        / "decode_logits_f32.bin"
    )

    hidden_array = np.fromfile(
        hidden_path,
        dtype=np.float32,
    )

    expected_logits = np.fromfile(
        expected_logits_path,
        dtype=np.float32,
    )

    if hidden_array.size != HIDDEN_SIZE:
        raise RuntimeError(
            "Layer 23输出元素数量错误："
            f"{hidden_array.size}"
        )

    if expected_logits.size != VOCAB_SIZE:
        raise RuntimeError(
            "参考logits元素数量错误："
            f"{expected_logits.size}"
        )

    print("===== Load model =====")

    load_start = time.perf_counter()

    model = (
        HunYuanVLForConditionalGeneration
        .from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .eval()
    )

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    final_norm, final_norm_path = (
        resolve_module(
            model,
            (
                "model.language_model.norm",
                "model.norm",
                "language_model.norm",
            ),
        )
    )

    lm_head, lm_head_path = (
        resolve_module(
            model,
            (
                "lm_head",
                "model.lm_head",
                "model.language_model.lm_head",
                "language_model.lm_head",
            ),
        )
    )

    hidden = (
        torch.from_numpy(
            hidden_array.copy()
        )
        .reshape(
            1,
            1,
            HIDDEN_SIZE,
        )
    )

    print(
        f"Model loaded in {load_seconds:.3f}s"
    )

    print(
        "Final norm module:",
        final_norm_path,
    )

    print(
        "LM Head module:",
        lm_head_path,
    )

    with torch.inference_mode():
        norm_output = final_norm(
            hidden
        )

        logits = lm_head(
            norm_output
        )

    norm_array = (
        norm_output.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
        .reshape(-1)
    )

    logits_array = (
        logits.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
        .reshape(-1)
    )

    if norm_array.size != HIDDEN_SIZE:
        raise RuntimeError(
            "Final RMSNorm输出数量错误："
            f"{norm_array.size}"
        )

    if logits_array.size != VOCAB_SIZE:
        raise RuntimeError(
            "LM Head输出数量错误："
            f"{logits_array.size}"
        )

    logits_error = maximum_abs_error(
        logits_array,
        expected_logits,
    )

    expected_token = int(
        expected_logits.argmax()
    )

    actual_token = int(
        logits_array.argmax()
    )

    print()
    print("===== Step 2 tail reference =====")

    print(
        "Final norm input shape:",
        tuple(hidden.shape),
    )

    print(
        "Final norm output shape:",
        tuple(norm_output.shape),
    )

    print(
        "Logits shape:",
        tuple(logits.shape),
    )

    print(
        "Logits maximum abs error:",
        f"{logits_error:.10e}",
    )

    print(
        "Expected token:",
        expected_token,
    )

    print(
        "Actual token:",
        actual_token,
    )

    if expected_token != EXPECTED_TOKEN:
        raise RuntimeError(
            "已有Step 2参考token不是206。"
        )

    if actual_token != EXPECTED_TOKEN:
        raise RuntimeError(
            "重新计算的Step 2 token不是206。"
        )

    if logits_error > 1.0e-5:
        raise RuntimeError(
            "重新计算的Step 2 logits"
            "与已有参考误差过大。"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hidden_array.tofile(
        OUTPUT_DIR
        / "final_norm_input_f32.bin"
    )

    norm_array.tofile(
        OUTPUT_DIR
        / "final_norm_output_f32.bin"
    )

    logits_array.tofile(
        OUTPUT_DIR
        / "decode_logits_f32.bin"
    )

    report = {
        "torch_version":
            torch.__version__,
        "torch_threads":
            TORCH_THREADS,
        "model_load_seconds":
            load_seconds,
        "final_norm_module":
            final_norm_path,
        "lm_head_module":
            lm_head_path,
        "hidden_size":
            HIDDEN_SIZE,
        "vocab_size":
            VOCAB_SIZE,
        "expected_token":
            expected_token,
        "actual_token":
            actual_token,
        "logits_maximum_abs_error":
            logits_error,
        "reference_directory":
            OUTPUT_DIR.relative_to(
                PROJECT_DIR
            ).as_posix(),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Report:", REPORT_PATH)

    print(
        "✅ Step 2 Final RMSNorm与LM Head参考捕获成功。"
    )


if __name__ == "__main__":
    main()
