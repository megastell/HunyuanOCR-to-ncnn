from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_DIR
    / "artifacts/decoder_layer0_decode_step2"
)

PNNX_SCRIPT = (
    MODEL_DIR
    / "decoder_layer0_decode_step2_pnnx.py"
)

REFERENCE_DIR = (
    PROJECT_DIR
    / "reference/smoke_en_cpu_fp32"
    / "decoder_layer0_decode_step2"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs"
    / "decoder_layer0_decode_step2_pnnx_parity.json"
)

TORCH_THREADS = 9


def load_tensor(
    name: str,
) -> torch.Tensor:
    path = REFERENCE_DIR / f"{name}.npy"

    if not path.is_file():
        raise FileNotFoundError(
            f"缺少参考张量：{path}"
        )

    array = np.load(path)

    return torch.from_numpy(array).float()


def load_generated_module() -> Any:
    if not PNNX_SCRIPT.is_file():
        raise FileNotFoundError(
            f"缺少PNNX生成脚本：{PNNX_SCRIPT}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            "decoder_layer0_decode_step2_pnnx",
            PNNX_SCRIPT,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "无法创建PNNX生成模块加载器。"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


def calculate_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | int]:
    actual_cpu = (
        actual.detach()
        .cpu()
        .float()
        .contiguous()
    )

    expected_cpu = (
        expected.detach()
        .cpu()
        .float()
        .contiguous()
    )

    if tuple(actual_cpu.shape) != tuple(
        expected_cpu.shape
    ):
        raise RuntimeError(
            "输出形状不一致："
            f"actual={tuple(actual_cpu.shape)}，"
            f"expected={tuple(expected_cpu.shape)}"
        )

    difference = (
        actual_cpu - expected_cpu
    ).abs()

    maximum_abs_error = float(
        difference.max().item()
    )

    mean_abs_error = float(
        difference.mean().item()
    )

    rmse = float(
        torch.sqrt(
            torch.mean(
                (
                    actual_cpu
                    - expected_cpu
                )
                ** 2
            )
        ).item()
    )

    actual_flat = actual_cpu.reshape(-1)
    expected_flat = expected_cpu.reshape(-1)

    denominator = float(
        (
            torch.linalg.vector_norm(
                actual_flat
            )
            * torch.linalg.vector_norm(
                expected_flat
            )
        ).item()
    )

    if denominator == 0.0:
        cosine_similarity = (
            1.0
            if torch.equal(
                actual_flat,
                expected_flat,
            )
            else 0.0
        )
    else:
        cosine_similarity = float(
            torch.dot(
                actual_flat,
                expected_flat,
            ).item()
            / denominator
        )

    cosine_similarity = min(
        1.0,
        max(-1.0, cosine_similarity),
    )

    maximum_index = int(
        difference.reshape(-1)
        .argmax()
        .item()
    )

    return {
        "maximum_abs_error":
            maximum_abs_error,
        "mean_abs_error":
            mean_abs_error,
        "rmse":
            rmse,
        "cosine_similarity":
            cosine_similarity,
        "maximum_error_index":
            maximum_index,
    }


def print_metrics(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    metrics: dict[str, float | int],
) -> None:
    print()
    print(name)
    print(
        "  actual shape:       ",
        tuple(actual.shape),
    )
    print(
        "  expected shape:     ",
        tuple(expected.shape),
    )
    print(
        "  maximum abs error:  "
        f"{metrics['maximum_abs_error']:.10e}"
    )
    print(
        "  mean abs error:     "
        f"{metrics['mean_abs_error']:.10e}"
    )
    print(
        "  RMSE:               "
        f"{metrics['rmse']:.10e}"
    )
    print(
        "  cosine similarity:  "
        f"{metrics['cosine_similarity']:.12f}"
    )


def main() -> None:
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    inputs = (
        load_tensor(
            "layer0_hidden_states"
        ),
        load_tensor(
            "layer0_attention_mask"
        ),
        load_tensor(
            "layer0_position_embeddings_0"
        ),
        load_tensor(
            "layer0_position_embeddings_1"
        ),
        load_tensor("past_key"),
        load_tensor("past_value"),
    )

    expected_outputs = (
        load_tensor("layer0_output"),
        load_tensor("present_key"),
        load_tensor("present_value"),
    )

    expected_input_shapes = (
        (1, 1, 1024),
        (1, 1, 1, 315),
        (4, 1, 1, 128),
        (4, 1, 1, 128),
        (1, 8, 314, 128),
        (1, 8, 314, 128),
    )

    expected_output_shapes = (
        (1, 1, 1024),
        (1, 8, 315, 128),
        (1, 8, 315, 128),
    )

    for index, (
        value,
        expected_shape,
    ) in enumerate(
        zip(
            inputs,
            expected_input_shapes,
        )
    ):
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"输入{index}形状错误："
                f"{tuple(value.shape)}，"
                f"预期={expected_shape}"
            )

    for index, (
        value,
        expected_shape,
    ) in enumerate(
        zip(
            expected_outputs,
            expected_output_shapes,
        )
    ):
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"参考输出{index}形状错误："
                f"{tuple(value.shape)}，"
                f"预期={expected_shape}"
            )

    print(
        "===== Load generated PNNX model ====="
    )

    generated_module = (
        load_generated_module()
    )

    model_class = getattr(
        generated_module,
        "Model",
        None,
    )

    if model_class is None:
        raise RuntimeError(
            "PNNX生成模块中不存在Model类。"
        )

    original_directory = Path.cwd()

    try:
        os.chdir(MODEL_DIR)

        model = model_class().eval()

        print(
            "Model class:",
            model_class.__name__,
        )

        print(
            "PNNX working directory:",
            Path.cwd(),
        )

        print()
        print(
            "===== PNNX Step 2 reference inference ====="
        )

        with torch.inference_mode():
            outputs = model(*inputs)
    finally:
        os.chdir(original_directory)

    if not isinstance(
        outputs,
        (tuple, list),
    ):
        raise RuntimeError(
            "PNNX模型没有返回tuple/list："
            f"{type(outputs).__name__}"
        )

    if len(outputs) != 3:
        raise RuntimeError(
            "PNNX输出数量错误："
            f"actual={len(outputs)}，"
            "expected=3"
        )

    output_names = (
        "layer_output",
        "present_key",
        "present_value",
    )

    thresholds = (
        (1.0e-5, 1.0e-7),
        (1.0e-6, 1.0e-8),
        (1.0e-6, 1.0e-8),
    )

    metrics_report: dict[
        str,
        dict[str, float | int],
    ] = {}

    for (
        name,
        actual,
        expected,
        threshold,
    ) in zip(
        output_names,
        outputs,
        expected_outputs,
        thresholds,
    ):
        metrics = calculate_metrics(
            actual,
            expected,
        )

        print_metrics(
            name,
            actual,
            expected,
            metrics,
        )

        for metric_name in (
            "maximum_abs_error",
            "mean_abs_error",
            "rmse",
            "cosine_similarity",
        ):
            metric_value = float(
                metrics[metric_name]
            )

            if not math.isfinite(metric_value):
                raise RuntimeError(
                    f"{name}的{metric_name}不是有限值。"
                )

        if (
            metrics["maximum_abs_error"]
            > threshold[0]
            or metrics["mean_abs_error"]
            > threshold[1]
            or metrics["cosine_similarity"]
            < 0.99999999
        ):
            raise RuntimeError(
                f"{name}的PNNX数值误差超限。"
            )

        metrics_report[name] = metrics

    report = {
        "decode_name":
            "decoder_layer0_decode_step2",
        "decode_step": 2,
        "torch_version":
            torch.__version__,
        "torch_threads":
            TORCH_THREADS,
        "pnnx_script":
            PNNX_SCRIPT.relative_to(
                PROJECT_DIR
            ).as_posix(),
        "input_shapes": [
            list(value.shape)
            for value in inputs
        ],
        "output_shapes": [
            list(value.shape)
            for value in outputs
        ],
        "metrics":
            metrics_report,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        "✅ Decoder Layer 0 Decode Step 2 "
        "PNNX三输出数值对齐成功。"
    )


if __name__ == "__main__":
    main()
