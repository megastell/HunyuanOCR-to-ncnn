from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import HunYuanVLForConditionalGeneration


PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

MODEL_DIR = (
    Path.home()
    / "work/hunyuanocr/models/HunyuanOCR-1.5"
)

REFERENCE_DIR: Path
OUTPUT_DIR: Path
REPORT_PATH: Path
LOG_PATH: Path

TORCH_THREADS = 9

BATCH_SIZE = 1
CURRENT_LENGTH = 1
PAST_LENGTH = 314
PRESENT_LENGTH = 315

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584

QUERY_HEADS = 16
KEY_VALUE_HEADS = 8
KEY_VALUE_GROUPS = 2
HEAD_DIM = 128

ROPE_COMPONENTS = 4
ROPE_SECTION_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the second single-token decode graph "
            "for a selected decoder layer."
        )
    )

    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Decoder layer index. Default: 0.",
    )

    return parser.parse_args()


def load_reference(
    name: str,
) -> torch.Tensor:
    path = REFERENCE_DIR / f"{name}.npy"

    if not path.is_file():
        raise FileNotFoundError(path)

    array = np.load(path)

    return torch.from_numpy(array)


def tensor_summary(
    value: torch.Tensor,
) -> dict[str, Any]:
    value = (
        value.detach()
        .cpu()
        .float()
        .contiguous()
    )

    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "minimum": float(value.min().item()),
        "maximum": float(value.max().item()),
        "mean": float(value.mean().item()),
    }


def calculate_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    # 指标统计使用float64，避免大张量的float32点积和
    # 范数累加使余弦相似度略小于或略大于1。
    # 这只影响统计精度，不改变模型输出。
    actual = (
        actual.detach()
        .cpu()
        .double()
        .reshape(-1)
    )

    expected = (
        expected.detach()
        .cpu()
        .double()
        .reshape(-1)
    )

    if actual.shape != expected.shape:
        raise RuntimeError(
            "比较张量形状不一致："
            f"{tuple(actual.shape)} 与 "
            f"{tuple(expected.shape)}"
        )

    difference = actual - expected
    absolute_difference = difference.abs()

    maximum_abs_error = float(
        absolute_difference.max().item()
    )

    mean_abs_error = float(
        absolute_difference.mean().item()
    )

    rmse = float(
        torch.sqrt(
            torch.mean(
                difference * difference
            )
        ).item()
    )

    denominator = (
        torch.linalg.vector_norm(actual)
        * torch.linalg.vector_norm(expected)
    )

    if float(denominator.item()) == 0.0:
        cosine_similarity = (
            1.0
            if torch.equal(actual, expected)
            else 0.0
        )
    else:
        raw_cosine = (
            torch.dot(actual, expected)
            / denominator
        )

        # 理论余弦值属于[-1, 1]；钳位仅消除最后几个
        # 浮点舍入位可能产生的微小越界。
        cosine_similarity = float(
            torch.clamp(
                raw_cosine,
                min=-1.0,
                max=1.0,
            ).item()
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
    }


def print_metrics(
    name: str,
    metrics: dict[str, float],
) -> None:
    print(name)

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


class DecoderLayerDecodeWrapper(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
    ) -> None:
        super().__init__()

        attention = layer.self_attn
        mlp = layer.mlp

        self.input_layernorm = (
            layer.input_layernorm
        )

        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj

        self.query_layernorm = (
            attention.query_layernorm
        )

        self.key_layernorm = (
            attention.key_layernorm
        )

        self.post_attention_layernorm = (
            layer.post_attention_layernorm
        )

        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj

        scaling = getattr(
            attention,
            "scaling",
            HEAD_DIM ** -0.5,
        )

        self.scaling = float(scaling)

    @staticmethod
    def merge_mrope_components(
        value: torch.Tensor,
    ) -> torch.Tensor:
        # 输入：
        # [4, 1, 1, 128]
        #
        # 最后一维按：
        # [32, 32, 32, 32]
        # 切成四段。
        #
        # 第0段取位置轴0，
        # 第1段取位置轴1，
        # 第2段取位置轴2，
        # 第3段取位置轴3。
        sections = torch.split(
            value,
            ROPE_SECTION_SIZE,
            dim=-1,
        )

        merged = torch.cat(
            (
                sections[0][0],
                sections[1][1],
                sections[2][2],
                sections[3][3],
            ),
            dim=-1,
        )

        return merged

    @staticmethod
    def rotate_half(
        value: torch.Tensor,
    ) -> torch.Tensor:
        first_half = value[..., :HEAD_DIM // 2]
        second_half = value[..., HEAD_DIM // 2:]

        return torch.cat(
            (
                -second_half,
                first_half,
            ),
            dim=-1,
        )

    @classmethod
    def apply_rotary(
        cls,
        value: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        return (
            value * cosine
            + cls.rotate_half(value) * sine
        )

    @staticmethod
    def repeat_key_value(
        value: torch.Tensor,
    ) -> torch.Tensor:
        # [1, 8, 315, 128]
        #   ↓
        # [1, 8, 2, 315, 128]
        #   ↓
        # [1, 16, 315, 128]
        value = value.unsqueeze(2)

        value = value.expand(
            BATCH_SIZE,
            KEY_VALUE_HEADS,
            KEY_VALUE_GROUPS,
            PRESENT_LENGTH,
            HEAD_DIM,
        )

        return value.reshape(
            BATCH_SIZE,
            QUERY_HEADS,
            PRESENT_LENGTH,
            HEAD_DIM,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        residual = hidden_states

        normalized_states = (
            self.input_layernorm(
                hidden_states
            )
        )

        query_states = self.q_proj(
            normalized_states
        )

        key_states = self.k_proj(
            normalized_states
        )

        value_states = self.v_proj(
            normalized_states
        )

        query_states = (
            query_states
            .reshape(
                BATCH_SIZE,
                CURRENT_LENGTH,
                QUERY_HEADS,
                HEAD_DIM,
            )
            .transpose(1, 2)
        )

        key_states = (
            key_states
            .reshape(
                BATCH_SIZE,
                CURRENT_LENGTH,
                KEY_VALUE_HEADS,
                HEAD_DIM,
            )
            .transpose(1, 2)
        )

        value_states = (
            value_states
            .reshape(
                BATCH_SIZE,
                CURRENT_LENGTH,
                KEY_VALUE_HEADS,
                HEAD_DIM,
            )
            .transpose(1, 2)
        )

        merged_cos = (
            self.merge_mrope_components(
                rope_cos
            )
        )

        merged_sin = (
            self.merge_mrope_components(
                rope_sin
            )
        )

        query_states = self.apply_rotary(
            query_states,
            merged_cos,
            merged_sin,
        )

        key_states = self.apply_rotary(
            key_states,
            merged_cos,
            merged_sin,
        )

        query_states = (
            self.query_layernorm(
                query_states
            )
        )

        key_states = (
            self.key_layernorm(
                key_states
            )
        )

        present_key = torch.cat(
            (
                past_key,
                key_states,
            ),
            dim=2,
        )

        present_value = torch.cat(
            (
                past_value,
                value_states,
            ),
            dim=2,
        )

        repeated_key = (
            self.repeat_key_value(
                present_key
            )
        )

        repeated_value = (
            self.repeat_key_value(
                present_value
            )
        )

        attention_scores = torch.matmul(
            query_states,
            repeated_key.transpose(
                -2,
                -1,
            ),
        )

        attention_scores = (
            attention_scores
            * self.scaling
        )

        attention_scores = (
            attention_scores
            + attention_mask
        )

        attention_probabilities = F.softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        attention_output = torch.matmul(
            attention_probabilities,
            repeated_value,
        )

        attention_output = (
            attention_output
            .transpose(1, 2)
            .contiguous()
            .reshape(
                BATCH_SIZE,
                CURRENT_LENGTH,
                QUERY_HEADS * HEAD_DIM,
            )
        )

        attention_output = self.o_proj(
            attention_output
        )

        hidden_states = (
            residual + attention_output
        )

        residual = hidden_states

        normalized_states = (
            self.post_attention_layernorm(
                hidden_states
            )
        )

        gate_output = self.gate_proj(
            normalized_states
        )

        up_output = self.up_proj(
            normalized_states
        )

        mlp_output = self.down_proj(
            F.silu(gate_output)
            * up_output
        )

        hidden_states = (
            residual + mlp_output
        )

        return (
            hidden_states,
            present_key,
            present_value,
        )


def validate_shapes(
    inputs: tuple[torch.Tensor, ...],
    expected_outputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    expected_input_shapes = (
        (
            BATCH_SIZE,
            CURRENT_LENGTH,
            HIDDEN_SIZE,
        ),
        (
            BATCH_SIZE,
            1,
            CURRENT_LENGTH,
            PRESENT_LENGTH,
        ),
        (
            ROPE_COMPONENTS,
            BATCH_SIZE,
            CURRENT_LENGTH,
            HEAD_DIM,
        ),
        (
            ROPE_COMPONENTS,
            BATCH_SIZE,
            CURRENT_LENGTH,
            HEAD_DIM,
        ),
        (
            BATCH_SIZE,
            KEY_VALUE_HEADS,
            PAST_LENGTH,
            HEAD_DIM,
        ),
        (
            BATCH_SIZE,
            KEY_VALUE_HEADS,
            PAST_LENGTH,
            HEAD_DIM,
        ),
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
                f"实际={tuple(value.shape)}，"
                f"预期={expected_shape}"
            )

    expected_output_shapes = (
        (
            BATCH_SIZE,
            CURRENT_LENGTH,
            HIDDEN_SIZE,
        ),
        (
            BATCH_SIZE,
            KEY_VALUE_HEADS,
            PRESENT_LENGTH,
            HEAD_DIM,
        ),
        (
            BATCH_SIZE,
            KEY_VALUE_HEADS,
            PRESENT_LENGTH,
            HEAD_DIM,
        ),
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
                f"输出{index}形状错误："
                f"实际={tuple(value.shape)}，"
                f"预期={expected_shape}"
            )


def validate_metrics(
    name: str,
    metrics: dict[str, float],
    maximum_tolerance: float,
    mean_tolerance: float,
) -> None:
    if (
        metrics["maximum_abs_error"]
        > maximum_tolerance
    ):
        raise RuntimeError(
            f"{name}最大误差超限："
            f"{metrics['maximum_abs_error']}"
        )

    if (
        metrics["mean_abs_error"]
        > mean_tolerance
    ):
        raise RuntimeError(
            f"{name}平均误差超限："
            f"{metrics['mean_abs_error']}"
        )

    if (
        metrics["cosine_similarity"]
        < 0.999999999
    ):
        raise RuntimeError(
            f"{name}余弦相似度不足："
            f"{metrics['cosine_similarity']}"
        )


def main() -> None:
    args = parse_args()
    layer_index = args.layer_index

    if layer_index < 0:
        raise ValueError(
            f"layer_index不能为负数：{layer_index}"
        )

    layer_prefix = f"layer{layer_index}"
    decode_name = (
        f"decoder_layer{layer_index}_decode_step2"
    )

    global REFERENCE_DIR
    global OUTPUT_DIR
    global REPORT_PATH
    global LOG_PATH

    REFERENCE_DIR = (
        PROJECT_DIR
        / "reference/smoke_en_cpu_fp32"
        / decode_name
    )

    OUTPUT_DIR = (
        PROJECT_DIR
        / "artifacts"
        / decode_name
    )

    REPORT_PATH = (
        PROJECT_DIR
        / "docs"
        / f"{decode_name}_export.json"
    )

    LOG_PATH = (
        PROJECT_DIR
        / "docs"
        / f"{decode_name}_export.txt"
    )

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hidden_states = load_reference(
        f"{layer_prefix}_hidden_states"
    ).float()

    attention_mask = load_reference(
        f"{layer_prefix}_attention_mask"
    ).float()

    rope_cos = load_reference(
        f"{layer_prefix}_position_embeddings_0"
    ).float()

    rope_sin = load_reference(
        f"{layer_prefix}_position_embeddings_1"
    ).float()

    past_key = load_reference(
        "past_key"
    ).float()

    past_value = load_reference(
        "past_value"
    ).float()

    expected_layer_output = (
        load_reference(
            f"{layer_prefix}_output"
        ).float()
    )

    expected_present_key = (
        load_reference(
            "present_key"
        ).float()
    )

    expected_present_value = (
        load_reference(
            "present_value"
        ).float()
    )

    inputs = (
        hidden_states,
        attention_mask,
        rope_cos,
        rope_sin,
        past_key,
        past_value,
    )

    expected_outputs = (
        expected_layer_output,
        expected_present_key,
        expected_present_value,
    )

    validate_shapes(
        inputs,
        expected_outputs,
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

    decoder_layers = (
        model.model
        .language_model
        .layers
    )

    if layer_index >= len(decoder_layers):
        raise ValueError(
            f"layer_index={layer_index}越界，"
            f"模型共有{len(decoder_layers)}层。"
        )

    decoder_layer = (
        decoder_layers[layer_index]
    )

    wrapper = (
        DecoderLayerDecodeWrapper(
            decoder_layer
        )
        .eval()
    )

    # wrapper已经持有所选Decoder层需要的子模块，
    # 可以释放完整模型其余部分。
    del decoder_layer
    del model

    gc.collect()

    print(
        f"Model loaded in {load_seconds:.3f}s"
    )

    print()
    print("===== Eager wrapper parity =====")

    with torch.inference_mode():
        eager_outputs = wrapper(*inputs)

    eager_metrics = {
        "layer_output":
            calculate_metrics(
                eager_outputs[0],
                expected_layer_output,
            ),

        "present_key":
            calculate_metrics(
                eager_outputs[1],
                expected_present_key,
            ),

        "present_value":
            calculate_metrics(
                eager_outputs[2],
                expected_present_value,
            ),
    }

    for name, metrics in (
        eager_metrics.items()
    ):
        print_metrics(name, metrics)

    validate_metrics(
        "Eager layer output",
        eager_metrics["layer_output"],
        maximum_tolerance=1e-5,
        mean_tolerance=1e-7,
    )

    validate_metrics(
        "Eager present key",
        eager_metrics["present_key"],
        maximum_tolerance=1e-6,
        mean_tolerance=1e-8,
    )

    validate_metrics(
        "Eager present value",
        eager_metrics["present_value"],
        maximum_tolerance=1e-6,
        mean_tolerance=1e-8,
    )

    print()
    print("===== TorchScript trace =====")

    trace_start = time.perf_counter()

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            inputs,
            strict=True,
            check_trace=True,
        )

        traced = torch.jit.freeze(
            traced.eval()
        )

    trace_seconds = (
        time.perf_counter()
        - trace_start
    )

    script_path = (
        OUTPUT_DIR
        / f"{decode_name}.pt"
    )

    traced.save(str(script_path))

    with torch.inference_mode():
        traced_outputs = traced(*inputs)

    traced_metrics = {
        "layer_output":
            calculate_metrics(
                traced_outputs[0],
                expected_layer_output,
            ),

        "present_key":
            calculate_metrics(
                traced_outputs[1],
                expected_present_key,
            ),

        "present_value":
            calculate_metrics(
                traced_outputs[2],
                expected_present_value,
            ),
    }

    print(
        f"Trace completed in "
        f"{trace_seconds:.3f}s"
    )

    print(
        "TorchScript:",
        script_path,
    )

    print(
        "TorchScript size:",
        f"{script_path.stat().st_size / 1024 / 1024:.2f} MiB",
    )

    for name, metrics in (
        traced_metrics.items()
    ):
        print_metrics(
            f"TorchScript {name}",
            metrics,
        )

    validate_metrics(
        "TorchScript layer output",
        traced_metrics["layer_output"],
        maximum_tolerance=1e-5,
        mean_tolerance=1e-7,
    )

    validate_metrics(
        "TorchScript present key",
        traced_metrics["present_key"],
        maximum_tolerance=1e-6,
        mean_tolerance=1e-8,
    )

    validate_metrics(
        "TorchScript present value",
        traced_metrics["present_value"],
        maximum_tolerance=1e-6,
        mean_tolerance=1e-8,
    )

    graph_text = str(
        traced.inlined_graph
    )

    if "prim::PythonOp" in graph_text:
        raise RuntimeError(
            "TorchScript图中仍包含prim::PythonOp。"
        )

    report = {
        "layer_index": layer_index,
        "decode_name": decode_name,
        "model_revision": (
            PROJECT_DIR
            / "configs/model_revision.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),

        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,

        "contract": {
            "inputs": {
                "hidden_states":
                    [1, 1, 1024],

                "attention_mask":
                    [1, 1, 1, 314],

                "rope_cos":
                    [4, 1, 1, 128],

                "rope_sin":
                    [4, 1, 1, 128],

                "past_key":
                    [1, 8, 313, 128],

                "past_value":
                    [1, 8, 313, 128],
            },

            "outputs": {
                "layer_output":
                    [1, 1, 1024],

                "present_key":
                    [1, 8, 314, 128],

                "present_value":
                    [1, 8, 314, 128],
            },
        },

        "model_load_seconds":
            load_seconds,

        "trace_seconds":
            trace_seconds,

        "torchscript_path":
            script_path.relative_to(
                PROJECT_DIR
            ).as_posix(),

        "torchscript_bytes":
            script_path.stat().st_size,

        "inputs": {
            "hidden_states":
                tensor_summary(hidden_states),

            "attention_mask":
                tensor_summary(attention_mask),

            "rope_cos":
                tensor_summary(rope_cos),

            "rope_sin":
                tensor_summary(rope_sin),

            "past_key":
                tensor_summary(past_key),

            "past_value":
                tensor_summary(past_value),
        },

        "eager_metrics":
            eager_metrics,

        "torchscript_metrics":
            traced_metrics,

        "graph_checks": {
            "contains_python_op":
                "prim::PythonOp"
                in graph_text,

            "contains_softmax":
                "aten::softmax"
                in graph_text,

            "contains_matmul":
                "aten::matmul"
                in graph_text,

            "contains_cat":
                "aten::cat"
                in graph_text,
        },
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
        f"✅ Decoder Layer {layer_index} Decode Step 2 "
        "Eager与TorchScript对齐成功。"
    )


if __name__ == "__main__":
    main()
