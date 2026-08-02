from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]

PNNX_SCRIPT = (
    PROJECT_DIR
    / "artifacts/decoder_layer0_decode"
    / "decoder_layer0_decode_pnnx.py"
)

REFERENCE_DIR = (
    PROJECT_DIR
    / "reference/smoke_en_cpu_fp32"
    / "decoder_layer0_decode"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/decoder_layer0_decode_pnnx_parity.json"
)

TORCH_THREADS = 9


def load_tensor(name: str) -> torch.Tensor:
    path = REFERENCE_DIR / f"{name}.npy"

    if not path.is_file():
        raise FileNotFoundError(
            f"缺少参考张量：{path}"
        )

    return torch.from_numpy(
        np.load(path)
    ).float()


def load_generated_module() -> Any:
    if not PNNX_SCRIPT.is_file():
        raise FileNotFoundError(
            f"缺少PNNX生成脚本：{PNNX_SCRIPT}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            "decoder_layer0_decode_generated_pnnx",
            PNNX_SCRIPT,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "无法创建PNNX模块加载规范。"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


def find_model_class(
    module: Any,
) -> type[torch.nn.Module]:
    direct_model = getattr(
        module,
        "Model",
        None,
    )

    if (
        isinstance(direct_model, type)
        and issubclass(
            direct_model,
            torch.nn.Module,
        )
    ):
        return direct_model

    candidates: list[type[torch.nn.Module]] = []

    for value in vars(module).values():
        if (
            isinstance(value, type)
            and value is not torch.nn.Module
            and issubclass(
                value,
                torch.nn.Module,
            )
            and value.__module__
            == module.__name__
        ):
            candidates.append(value)

    if len(candidates) != 1:
        raise RuntimeError(
            "无法唯一确定PNNX模型类："
            f"{[item.__name__ for item in candidates]}"
        )

    return candidates[0]


def calculate_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | int]:
    if actual.shape != expected.shape:
        raise RuntimeError(
            "张量形状不一致："
            f"actual={tuple(actual.shape)}，"
            f"expected={tuple(expected.shape)}"
        )

    actual_flat = (
        actual.detach()
        .cpu()
        .double()
        .reshape(-1)
    )

    expected_flat = (
        expected.detach()
        .cpu()
        .double()
        .reshape(-1)
    )

    difference = actual_flat - expected_flat
    absolute_difference = difference.abs()

    max_index = int(
        absolute_difference.argmax().item()
    )

    denominator = (
        torch.linalg.vector_norm(actual_flat)
        * torch.linalg.vector_norm(expected_flat)
    )

    if denominator.item() == 0.0:
        cosine_similarity = (
            1.0
            if torch.equal(
                actual_flat,
                expected_flat,
            )
            else 0.0
        )
    else:
        raw_cosine = (
            torch.dot(
                actual_flat,
                expected_flat,
            )
            / denominator
        )

        cosine_similarity = float(
            torch.clamp(
                raw_cosine,
                min=-1.0,
                max=1.0,
            ).item()
        )

    return {
        "maximum_abs_error": float(
            absolute_difference.max().item()
        ),
        "mean_abs_error": float(
            absolute_difference.mean().item()
        ),
        "rmse": float(
            torch.sqrt(
                torch.mean(
                    difference * difference
                )
            ).item()
        ),
        "cosine_similarity":
            cosine_similarity,
        "max_error_flat_index":
            max_index,
        "actual_at_max": float(
            actual_flat[max_index].item()
        ),
        "expected_at_max": float(
            expected_flat[max_index].item()
        ),
    }


def print_metrics(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    metrics: dict[str, float | int],
) -> None:
    print(name)
    print(
        "  actual shape:      ",
        tuple(actual.shape),
    )
    print(
        "  expected shape:    ",
        tuple(expected.shape),
    )
    print(
        "  maximum abs error: ",
        f"{metrics['maximum_abs_error']:.10e}",
    )
    print(
        "  mean abs error:    ",
        f"{metrics['mean_abs_error']:.10e}",
    )
    print(
        "  RMSE:              ",
        f"{metrics['rmse']:.10e}",
    )
    print(
        "  cosine similarity: ",
        f"{metrics['cosine_similarity']:.12f}",
    )


def validate_output(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    maximum_tolerance: float,
    mean_tolerance: float,
) -> dict[str, float | int]:
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

    if (
        float(metrics["maximum_abs_error"])
        > maximum_tolerance
    ):
        raise RuntimeError(
            f"{name}最大误差超限："
            f"{metrics['maximum_abs_error']} > "
            f"{maximum_tolerance}"
        )

    if (
        float(metrics["mean_abs_error"])
        > mean_tolerance
    ):
        raise RuntimeError(
            f"{name}平均误差超限："
            f"{metrics['mean_abs_error']} > "
            f"{mean_tolerance}"
        )

    if (
        float(metrics["cosine_similarity"])
        < 0.999999999
    ):
        raise RuntimeError(
            f"{name}余弦相似度不足："
            f"{metrics['cosine_similarity']}"
        )

    return metrics


def main() -> None:
    torch.set_grad_enabled(False)
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
        (1, 1, 1, 314),
        (4, 1, 1, 128),
        (4, 1, 1, 128),
        (1, 8, 313, 128),
        (1, 8, 313, 128),
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
                f"actual={tuple(value.shape)}，"
                f"expected={expected_shape}"
            )

    print(
        "===== Load generated PNNX model ====="
    )

    generated_module = load_generated_module()

    model_class = find_model_class(
        generated_module
    )

    # 生成脚本可能按照相对路径读取pnnx.bin，
    # 因此在它所在目录下初始化并执行。
    original_directory = Path.cwd()

    try:
        os.chdir(PROJECT_DIR)

        model = model_class().eval()

        print(
            "Model class:",
            model_class.__name__,
        )

        print()
        print(
            "===== PNNX reference inference ====="
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
        (1e-5, 1e-7),
        (1e-6, 1e-8),
        (1e-6, 1e-8),
    )

    metrics_report: dict[
        str,
        dict[str, float | int],
    ] = {}

    for (
        name,
        actual,
        expected,
        tolerances,
    ) in zip(
        output_names,
        outputs,
        expected_outputs,
        thresholds,
    ):
        print()

        metrics_report[name] = (
            validate_output(
                name,
                actual,
                expected,
                maximum_tolerance=
                    tolerances[0],
                mean_tolerance=
                    tolerances[1],
            )
        )

    report = {
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

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("Report:", REPORT_PATH)
    print(
        "✅ Decoder Layer 0 Decode "
        "PNNX三输出数值对齐成功。"
    )


if __name__ == "__main__":
    main()
