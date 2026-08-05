from __future__ import annotations

import json
import time
from pathlib import Path

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

STEP2_LAYER0_REFERENCE = (
    PROJECT_DIR
    / "artifacts/decoder_layer0_decode_step2/reference"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "artifacts/token_embedding_feedback/reference"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/token_embedding_feedback_reference.json"
)

TOKEN_ID = 5112
HIDDEN_SIZE = 1024
TORCH_THREADS = 9


def find_module_path(
    model: torch.nn.Module,
    target: torch.nn.Module,
) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name or "<root>"

    return "<unknown>"


def max_abs_error(
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    if actual.shape != expected.shape:
        raise RuntimeError(
            "数组形状不一致："
            f"{actual.shape} != {expected.shape}"
        )

    difference = (
        actual.astype(np.float64)
        - expected.astype(np.float64)
    )

    return float(
        np.max(
            np.abs(difference)
        )
    )


def mean_abs_error(
    actual: np.ndarray,
    expected: np.ndarray,
) -> float:
    difference = (
        actual.astype(np.float64)
        - expected.astype(np.float64)
    )

    return float(
        np.mean(
            np.abs(difference)
        )
    )


def main() -> None:
    logging.disable_progress_bar()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(
        TORCH_THREADS
    )

    reference_path = (
        STEP2_LAYER0_REFERENCE
        / "layer0_hidden_states_f32.bin"
    )

    reference_hidden = np.fromfile(
        reference_path,
        dtype=np.float32,
    )

    if reference_hidden.size != HIDDEN_SIZE:
        raise RuntimeError(
            "Step 2 Layer 0参考hidden数量错误："
            f"{reference_hidden.size}"
        )

    reference_hidden = (
        reference_hidden.reshape(
            1,
            1,
            HIDDEN_SIZE,
        )
    )

    print("===== Load model =====")

    start = time.perf_counter()

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
        - start
    )

    input_embedding = (
        model.get_input_embeddings()
    )

    if input_embedding is None:
        raise RuntimeError(
            "model.get_input_embeddings()返回None"
        )

    output_embedding = (
        model.get_output_embeddings()
    )

    input_embedding_path = (
        find_module_path(
            model,
            input_embedding,
        )
    )

    output_embedding_path = (
        (
            find_module_path(
                model,
                output_embedding,
            )
        )
        if output_embedding is not None
        else None
    )

    token_ids = torch.tensor(
        [[TOKEN_ID]],
        dtype=torch.long,
    )

    with torch.inference_mode():
        embedding_output = (
            input_embedding(
                token_ids
            )
        )

    actual_hidden = (
        embedding_output.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    if actual_hidden.shape != (
        1,
        1,
        HIDDEN_SIZE,
    ):
        raise RuntimeError(
            "Token Embedding输出形状错误："
            f"{actual_hidden.shape}"
        )

    embedding_row = (
        input_embedding.weight[
            TOKEN_ID
        ]
        .detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
        .reshape(
            1,
            1,
            HIDDEN_SIZE,
        )
    )

    embedding_vs_reference_max = (
        max_abs_error(
            actual_hidden,
            reference_hidden,
        )
    )

    embedding_vs_reference_mean = (
        mean_abs_error(
            actual_hidden,
            reference_hidden,
        )
    )

    output_vs_weight_row_max = (
        max_abs_error(
            actual_hidden,
            embedding_row,
        )
    )

    byte_identical = bool(
        np.array_equal(
            actual_hidden,
            reference_hidden,
        )
    )

    tied_storage = False
    lm_head_row_error = None

    if output_embedding is not None:
        if not hasattr(
            output_embedding,
            "weight",
        ):
            raise RuntimeError(
                "输出Embedding/LM Head没有weight"
            )

        tied_storage = bool(
            input_embedding.weight.data_ptr()
            == output_embedding.weight.data_ptr()
        )

        output_weight = (
            output_embedding.weight
        )

        if (
            output_weight.ndim == 2
            and output_weight.shape[0]
                > TOKEN_ID
            and output_weight.shape[1]
                == HIDDEN_SIZE
        ):
            lm_head_row = (
                output_weight[
                    TOKEN_ID
                ]
                .detach()
                .cpu()
                .contiguous()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
                .reshape(
                    1,
                    1,
                    HIDDEN_SIZE,
                )
            )

            lm_head_row_error = (
                max_abs_error(
                    embedding_row,
                    lm_head_row,
                )
            )

    print(
        f"Model loaded in {load_seconds:.3f}s"
    )

    print(
        "Input embedding module:",
        input_embedding_path,
    )

    print(
        "Output embedding module:",
        output_embedding_path,
    )

    print(
        "Input embedding weight shape:",
        tuple(
            input_embedding.weight.shape
        ),
    )

    print(
        "Input embedding dtype:",
        input_embedding.weight.dtype,
    )

    print(
        "Token ID:",
        TOKEN_ID,
    )

    print(
        "Embedding output shape:",
        tuple(actual_hidden.shape),
    )

    print(
        "Embedding vs Step 2 Layer 0 max error:",
        f"{embedding_vs_reference_max:.10e}",
    )

    print(
        "Embedding vs Step 2 Layer 0 mean error:",
        f"{embedding_vs_reference_mean:.10e}",
    )

    print(
        "Embedding output vs weight row max error:",
        f"{output_vs_weight_row_max:.10e}",
    )

    print(
        "Embedding and Step 2 hidden byte-identical:",
        byte_identical,
    )

    print(
        "Input embedding and LM Head share storage:",
        tied_storage,
    )

    print(
        "Embedding row vs LM Head row max error:",
        (
            f"{lm_head_row_error:.10e}"
            if lm_head_row_error is not None
            else "not checked"
        ),
    )

    if output_vs_weight_row_max != 0.0:
        raise RuntimeError(
            "Embedding输出与权重第5112行不一致。"
        )

    if embedding_vs_reference_max > 1.0e-7:
        raise RuntimeError(
            "Token 5112 Embedding与Step 2"
            " Layer 0参考hidden不一致。"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.asarray(
        [TOKEN_ID],
        dtype=np.int64,
    ).tofile(
        OUTPUT_DIR
        / "token_id_i64.bin"
    )

    actual_hidden.reshape(-1).tofile(
        OUTPUT_DIR
        / "token_5112_embedding_f32.bin"
    )

    report = {
        "token_id":
            TOKEN_ID,
        "hidden_size":
            HIDDEN_SIZE,
        "torch_threads":
            TORCH_THREADS,
        "torch_version":
            torch.__version__,
        "model_load_seconds":
            load_seconds,
        "input_embedding_module":
            input_embedding_path,
        "output_embedding_module":
            output_embedding_path,
        "input_embedding_weight_shape":
            list(
                input_embedding.weight.shape
            ),
        "input_embedding_dtype":
            str(
                input_embedding.weight.dtype
            ),
        "embedding_output_shape":
            list(actual_hidden.shape),
        "embedding_vs_step2_hidden": {
            "maximum_abs_error":
                embedding_vs_reference_max,
            "mean_abs_error":
                embedding_vs_reference_mean,
            "byte_identical":
                byte_identical,
        },
        "embedding_output_vs_weight_row": {
            "maximum_abs_error":
                output_vs_weight_row_max,
        },
        "lm_head_relationship": {
            "shares_weight_storage":
                tied_storage,
            "row_5112_maximum_abs_error":
                lm_head_row_error,
        },
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
        "✅ token 5112 Embedding反馈契约捕获成功。"
    )


if __name__ == "__main__":
    main()
