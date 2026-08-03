from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import HunYuanVLForConditionalGeneration


HOME_DIR = Path.home()

ROOT = (
    HOME_DIR
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

MODEL_DIR = (
    HOME_DIR
    / "work/hunyuanocr/models/HunyuanOCR-1.5"
)

LAYER23_OUTPUT_PATH = (
    ROOT
    / "artifacts/decoder_layer23_decode/reference"
    / "layer23_output_f32.bin"
)

EXPECTED_LOGITS_PATH = (
    ROOT
    / "artifacts/decoder_layer23_decode/reference"
    / "decode_logits_f32.bin"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts/decode_tail/reference"
)

REPORT_PATH = (
    ROOT
    / "docs/decode_tail_reference.json"
)

HIDDEN_SIZE = 1024
VOCAB_SIZE = 120818
THREADS = 9


def load_exact(
    path: Path,
    expected_count: int,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    values = np.fromfile(
        path,
        dtype=np.float32,
    )

    if values.size != expected_count:
        raise RuntimeError(
            f"{path}元素数量错误："
            f"实际{values.size}，预期{expected_count}"
        )

    return values


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


def calculate_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
) -> dict[str, float]:
    actual64 = actual.astype(
        np.float64,
        copy=False,
    )

    expected64 = expected.astype(
        np.float64,
        copy=False,
    )

    difference = actual64 - expected64
    absolute = np.abs(difference)

    denominator = (
        np.linalg.norm(actual64)
        * np.linalg.norm(expected64)
    )

    cosine = float(
        np.dot(
            actual64.reshape(-1),
            expected64.reshape(-1),
        )
        / denominator
    )

    return {
        "maximum_abs_error": float(
            absolute.max()
        ),
        "mean_abs_error": float(
            absolute.mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    difference * difference
                )
            )
        ),
        "cosine_similarity": cosine,
    }


def print_metrics(
    title: str,
    metrics: dict[str, float],
) -> None:
    print(title)
    print(
        "  maximum abs error:",
        f"{metrics['maximum_abs_error']:.10e}",
    )
    print(
        "  mean abs error:   ",
        f"{metrics['mean_abs_error']:.10e}",
    )
    print(
        "  RMSE:             ",
        f"{metrics['rmse']:.10e}",
    )
    print(
        "  cosine similarity:",
        f"{metrics['cosine_similarity']:.12f}",
    )


def main() -> None:
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(THREADS)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    layer23_output = load_exact(
        LAYER23_OUTPUT_PATH,
        HIDDEN_SIZE,
    )

    expected_logits = load_exact(
        EXPECTED_LOGITS_PATH,
        VOCAB_SIZE,
    )

    hidden_state = (
        torch.from_numpy(
            layer23_output.copy()
        )
        .reshape(1, 1, HIDDEN_SIZE)
        .contiguous()
    )

    print("===== Load model =====", flush=True)

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

    final_norm = (
        model.model.language_model.norm
    )

    lm_head = model.lm_head

    print(
        "Final norm class:",
        final_norm.__class__.__name__,
    )

    print(
        "Final norm weight:",
        tuple(final_norm.weight.shape),
    )

    print(
        "Final norm epsilon:",
        model.config.text_config.rms_norm_eps,
    )

    print(
        "LM Head weight:",
        tuple(lm_head.weight.shape),
    )

    print(
        "\n===== Single-token decode tail =====",
        flush=True,
    )

    with torch.inference_mode():
        final_norm_output = final_norm(
            hidden_state
        )

        actual_logits = lm_head(
            final_norm_output
        )

    if tuple(final_norm_output.shape) != (
        1,
        1,
        HIDDEN_SIZE,
    ):
        raise RuntimeError(
            "Final RMSNorm输出形状错误："
            f"{tuple(final_norm_output.shape)}"
        )

    if tuple(actual_logits.shape) != (
        1,
        1,
        VOCAB_SIZE,
    ):
        raise RuntimeError(
            "LM Head输出形状错误："
            f"{tuple(actual_logits.shape)}"
        )

    actual_logits_numpy = (
        actual_logits
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(
            np.float32,
            copy=False,
        )
    )

    expected_token = int(
        expected_logits.argmax()
    )

    actual_token = int(
        actual_logits_numpy.argmax()
    )

    metrics = calculate_metrics(
        actual_logits_numpy,
        expected_logits,
    )

    print("Expected decode token:", expected_token)
    print("Actual decode token:  ", actual_token)
    print()

    print_metrics(
        "Tail logits vs captured decode logits",
        metrics,
    )

    if actual_token != expected_token:
        raise RuntimeError(
            "Decode尾部argmax不一致"
        )

    if (
        metrics["maximum_abs_error"]
        > 1.0e-6
    ):
        raise RuntimeError(
            "Decode尾部logits与完整模型捕获结果不一致"
        )

    norm_input_path = (
        OUTPUT_DIR
        / "final_norm_input_f32.bin"
    )

    norm_output_path = (
        OUTPUT_DIR
        / "final_norm_output_f32.bin"
    )

    logits_path = (
        OUTPUT_DIR
        / "decode_logits_f32.bin"
    )

    hidden_state.numpy().astype(
        np.float32,
        copy=False,
    ).tofile(norm_input_path)

    final_norm_output.cpu().numpy().astype(
        np.float32,
        copy=False,
    ).tofile(norm_output_path)

    actual_logits_numpy.tofile(
        logits_path
    )

    report = {
        "source_layer23_output": str(
            LAYER23_OUTPUT_PATH
        ),
        "source_expected_logits": str(
            EXPECTED_LOGITS_PATH
        ),
        "final_norm_module": (
            "model.language_model.norm"
        ),
        "final_norm_class": (
            final_norm.__class__.__name__
        ),
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "epsilon": (
            model.config.text_config.rms_norm_eps
        ),
        "input_shape": [1, 1, HIDDEN_SIZE],
        "norm_output_shape": [
            1,
            1,
            HIDDEN_SIZE,
        ],
        "logits_shape": [
            1,
            1,
            VOCAB_SIZE,
        ],
        "expected_decode_token": expected_token,
        "actual_decode_token": actual_token,
        "logits_metrics": metrics,
        "final_norm_input": str(
            norm_input_path
        ),
        "final_norm_output": str(
            norm_output_path
        ),
        "decode_logits": str(
            logits_path
        ),
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
    print(
        "Final norm input:",
        norm_input_path,
    )

    print(
        "Final norm output:",
        norm_output_path,
    )

    print(
        "Decode logits:",
        logits_path,
    )

    print(
        "Report:",
        REPORT_PATH,
    )

    print()
    print(
        "✅ Single-token decode尾部"
        "PyTorch参考链捕获成功。"
    )


if __name__ == "__main__":
    main()
