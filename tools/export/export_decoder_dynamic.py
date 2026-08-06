from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import HunYuanVLForConditionalGeneration


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
PNNX_PATH = Path.home() / "work/hunyuanocr/.venv-pnnx/bin/pnnx"
BASE_EXPORT_PATH = (
    PROJECT_DIR / "tools/export/export_decoder_decode_step3.py"
)

TORCH_THREADS = 9
KEY_VALUE_GROUPS = 2

CASES = (
    ("step1", "decoder_layer{layer}_decode", 313, 314),
    ("step2", "decoder_layer{layer}_decode_step2", 314, 315),
    ("step3", "decoder_layer{layer}_decode_step3", 315, 316),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one decoder layer with a dynamic KV-cache length and "
            "validate the TorchScript graph on decode steps 1 through 3."
        )
    )
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument(
        "--skip-pnnx",
        action="store_true",
        help="Only generate and validate the dynamic TorchScript graph.",
    )
    return parser.parse_args()


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "decoder_dynamic_export_base",
        BASE_EXPORT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_EXPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_case(layer_index: int, template: str) -> tuple[torch.Tensor, ...]:
    name = template.format(layer=layer_index)
    reference_dir = PROJECT_DIR / "reference/smoke_en_cpu_fp32" / name
    prefix = f"layer{layer_index}"

    def load(stem: str) -> torch.Tensor:
        path = reference_dir / f"{stem}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        return torch.from_numpy(np.load(path)).float()

    return (
        load(f"{prefix}_hidden_states"),
        load(f"{prefix}_attention_mask"),
        load(f"{prefix}_position_embeddings_0"),
        load(f"{prefix}_position_embeddings_1"),
        load("past_key"),
        load("past_value"),
        load(f"{prefix}_output"),
        load("present_key"),
        load("present_value"),
    )


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual64 = actual.detach().cpu().double().reshape(-1)
    expected64 = expected.detach().cpu().double().reshape(-1)
    if actual64.shape != expected64.shape:
        raise RuntimeError(
            f"Shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    difference = (actual64 - expected64).abs()
    return {
        "maximum_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }


class DynamicDecoderLayer(nn.Module):
    def __init__(self, fixed_wrapper: nn.Module) -> None:
        super().__init__()
        self.fixed_wrapper = fixed_wrapper

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wrapper = self.fixed_wrapper
        residual = hidden_states
        normalized_states = wrapper.input_layernorm(hidden_states)

        query_states = wrapper.q_proj(normalized_states)
        key_states = wrapper.k_proj(normalized_states)
        value_states = wrapper.v_proj(normalized_states)

        query_states = query_states.reshape(1, 1, 16, 128).transpose(1, 2)
        key_states = key_states.reshape(1, 1, 8, 128).transpose(1, 2)
        value_states = value_states.reshape(1, 1, 8, 128).transpose(1, 2)

        merged_cos = wrapper.merge_mrope_components(rope_cos)
        merged_sin = wrapper.merge_mrope_components(rope_sin)
        query_states = wrapper.apply_rotary(
            query_states, merged_cos, merged_sin
        )
        key_states = wrapper.apply_rotary(key_states, merged_cos, merged_sin)
        query_states = wrapper.query_layernorm(query_states)
        key_states = wrapper.key_layernorm(key_states)

        present_key = torch.cat((past_key, key_states), dim=2)
        present_value = torch.cat((past_value, value_states), dim=2)

        # Flattening the KV-head and repeat axes keeps the sequence dimension
        # symbolic in TorchScript and lets pnnx resolve it from inputshape2.
        repeated_key = (
            present_key.unsqueeze(2)
            .expand(-1, -1, KEY_VALUE_GROUPS, -1, -1)
            .flatten(1, 2)
        )
        repeated_value = (
            present_value.unsqueeze(2)
            .expand(-1, -1, KEY_VALUE_GROUPS, -1, -1)
            .flatten(1, 2)
        )

        attention_scores = torch.matmul(
            query_states, repeated_key.transpose(-2, -1)
        )
        attention_scores = attention_scores * wrapper.scaling
        attention_scores = attention_scores + attention_mask
        attention_probabilities = torch.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attention_output = torch.matmul(
            attention_probabilities, repeated_value
        )
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(1, 1, 16 * 128)
        )
        hidden_states = residual + wrapper.o_proj(attention_output)

        residual = hidden_states
        normalized_states = wrapper.post_attention_layernorm(hidden_states)
        mlp_output = wrapper.down_proj(
            torch.nn.functional.silu(wrapper.gate_proj(normalized_states))
            * wrapper.up_proj(normalized_states)
        )
        hidden_states = residual + mlp_output
        return hidden_states, present_key, present_value


def shape_argument(values: tuple[torch.Tensor, ...]) -> str:
    return ",".join(
        "[" + ",".join(str(dimension) for dimension in value.shape) + "]"
        for value in values
    )


def main() -> None:
    args = parse_args()
    layer_index = args.layer_index
    if not 0 <= layer_index < 24:
        raise ValueError(f"layer-index must be in [0, 23], got {layer_index}")

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    case_data = {
        label: load_case(layer_index, template)
        for label, template, _, _ in CASES
    }
    output_dir = (
        PROJECT_DIR / "artifacts" / f"decoder_layer{layer_index}_decode_dynamic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        PROJECT_DIR / "docs" / f"decoder_layer{layer_index}_decode_dynamic.json"
    )
    pnnx_log_path = (
        PROJECT_DIR / "docs" / f"decoder_layer{layer_index}_decode_dynamic_pnnx.txt"
    )
    model_stem = f"decoder_layer{layer_index}_decode_dynamic"
    script_path = output_dir / f"{model_stem}.pt"

    base = load_base_module()
    load_start = time.perf_counter()
    model = (
        HunYuanVLForConditionalGeneration.from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .eval()
    )
    fixed_wrapper = base.DecoderLayerDecodeWrapper(
        model.model.language_model.layers[layer_index]
    ).eval()
    wrapper = DynamicDecoderLayer(fixed_wrapper).eval()
    del model
    gc.collect()

    trace_inputs = case_data["step1"][:6]
    check_inputs = [case_data["step2"][:6], case_data["step3"][:6]]
    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            trace_inputs,
            check_inputs=check_inputs,
            strict=True,
            check_trace=True,
        )
        traced = torch.jit.freeze(traced.eval())
    traced.save(str(script_path))

    validation: dict[str, Any] = {}
    with torch.inference_mode():
        for label, _, past_length, present_length in CASES:
            values = case_data[label]
            actual = traced(*values[:6])
            expected = values[6:]
            case_metrics = {
                "layer_output": metrics(actual[0], expected[0]),
                "present_key": metrics(actual[1], expected[1]),
                "present_value": metrics(actual[2], expected[2]),
            }
            for output_name, output_metrics in case_metrics.items():
                if output_metrics["maximum_abs_error"] > 1e-5:
                    raise RuntimeError(
                        f"{label} {output_name} failed: {output_metrics}"
                    )
            validation[label] = {
                "past_length": past_length,
                "present_length": present_length,
                "metrics": case_metrics,
            }

    pnnx_command: list[str] | None = None
    if not args.skip_pnnx:
        first_inputs = case_data["step1"][:6]
        second_inputs = case_data["step3"][:6]
        pnnx_command = [
            str(PNNX_PATH),
            str(script_path),
            f"inputshape={shape_argument(first_inputs)}",
            f"inputshape2={shape_argument(second_inputs)}",
            "fp16=0",
            "optlevel=2",
        ]
        conversion = subprocess.run(
            pnnx_command,
            cwd=output_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        pnnx_log_path.write_text(conversion.stdout, encoding="utf-8")
        if conversion.returncode != 0:
            raise RuntimeError(
                f"pnnx failed with status {conversion.returncode}; "
                f"see {pnnx_log_path}"
            )

    report = {
        "layer_index": layer_index,
        "model_revision": (
            PROJECT_DIR / "configs/model_revision.txt"
        ).read_text(encoding="utf-8").strip(),
        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,
        "model_load_and_trace_seconds": time.perf_counter() - load_start,
        "torchscript_path": script_path.relative_to(PROJECT_DIR).as_posix(),
        "torchscript_bytes": script_path.stat().st_size,
        "validation": validation,
        "pnnx_command": pnnx_command,
        "pnnx_log": (
            pnnx_log_path.relative_to(PROJECT_DIR).as_posix()
            if pnnx_command is not None
            else None
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"Dynamic TorchScript: {script_path}")
    print(f"Report: {report_path}")
    for label, case_report in validation.items():
        maximum = max(
            item["maximum_abs_error"]
            for item in case_report["metrics"].values()
        )
        print(
            f"{label}: {case_report['past_length']} -> "
            f"{case_report['present_length']}, max abs error={maximum:.10e}"
        )
    if pnnx_command is not None:
        print(f"pnnx log: {pnnx_log_path}")


if __name__ == "__main__":
    main()
