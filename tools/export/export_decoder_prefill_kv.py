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
from transformers.utils import logging


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
PNNX_PATH = Path.home() / "work/hunyuanocr/.venv-pnnx/bin/pnnx"
BASE_EXPORT_PATH = PROJECT_DIR / "tools/export/export_decoder_decode_step3.py"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DOCS_DIR = PROJECT_DIR / "docs"
REFERENCE_ROOT = PROJECT_DIR / "reference/smoke_en_cpu_fp32"

TORCH_THREADS = 9
LAYER_COUNT = 24
BATCH_SIZE = 1
SEQUENCE_LENGTH = 313
ALTERNATE_SEQUENCE_LENGTH = 299
HIDDEN_SIZE = 1024
QUERY_HEADS = 16
KV_HEADS = 8
KV_GROUPS = 2
HEAD_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export HunyuanOCR decoder prefill layers with hidden, key, and "
            "value outputs. All 24 layers are exported when no index is given."
        )
    )
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--pnnx", type=Path, default=PNNX_PATH)
    parser.add_argument("--skip-pnnx", action="store_true")
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load Python module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_f32(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.from_numpy(np.load(path)).float().contiguous()
    return value


def load_case(layer_index: int) -> tuple[torch.Tensor, ...]:
    layer_dir = REFERENCE_ROOT / f"decoder_layer{layer_index}_prefill_kv"
    shared_dir = REFERENCE_ROOT / "decoder_layer0_prefill_kv"
    prefix = f"layer{layer_index}"
    return (
        load_f32(layer_dir / f"{prefix}_hidden_states.npy"),
        load_f32(shared_dir / "layer0_attention_mask.npy"),
        load_f32(shared_dir / "layer0_position_embeddings_0.npy"),
        load_f32(shared_dir / "layer0_position_embeddings_1.npy"),
        load_f32(layer_dir / f"{prefix}_output.npy"),
        load_f32(layer_dir / "present_key.npy"),
        load_f32(layer_dir / "present_value.npy"),
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


class PrefillDecoderLayer(nn.Module):
    def __init__(self, fixed_wrapper: nn.Module) -> None:
        super().__init__()
        self.fixed_wrapper = fixed_wrapper

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wrapper = self.fixed_wrapper
        sequence_length = hidden_states.shape[1]
        residual = hidden_states
        normalized_states = wrapper.input_layernorm(hidden_states)

        query_states = wrapper.q_proj(normalized_states)
        key_states = wrapper.k_proj(normalized_states)
        value_states = wrapper.v_proj(normalized_states)

        query_states = query_states.reshape(
            BATCH_SIZE, sequence_length, QUERY_HEADS, HEAD_DIM
        ).transpose(1, 2)
        key_states = key_states.reshape(
            BATCH_SIZE, sequence_length, KV_HEADS, HEAD_DIM
        ).transpose(1, 2)
        value_states = value_states.reshape(
            BATCH_SIZE, sequence_length, KV_HEADS, HEAD_DIM
        ).transpose(1, 2)

        merged_cos = wrapper.merge_mrope_components(rope_cos)
        merged_sin = wrapper.merge_mrope_components(rope_sin)
        query_states = wrapper.apply_rotary(query_states, merged_cos, merged_sin)
        key_states = wrapper.apply_rotary(key_states, merged_cos, merged_sin)
        query_states = wrapper.query_layernorm(query_states)
        present_key = wrapper.key_layernorm(key_states)
        present_value = value_states

        repeated_key = (
            present_key.unsqueeze(2)
            .expand(-1, -1, KV_GROUPS, -1, -1)
            .flatten(1, 2)
        )
        repeated_value = (
            present_value.unsqueeze(2)
            .expand(-1, -1, KV_GROUPS, -1, -1)
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
        attention_output = torch.matmul(attention_probabilities, repeated_value)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(BATCH_SIZE, sequence_length, QUERY_HEADS * HEAD_DIM)
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


def export_layer(
    layer_index: int,
    model: HunYuanVLForConditionalGeneration,
    base: Any,
    skip_pnnx: bool,
) -> dict[str, Any]:
    layer_start = time.perf_counter()
    values = load_case(layer_index)
    inputs = values[:4]
    expected = values[4:]

    fixed_wrapper = base.DecoderLayerDecodeWrapper(
        model.model.language_model.layers[layer_index]
    ).eval()
    wrapper = PrefillDecoderLayer(fixed_wrapper).eval()

    with torch.inference_mode():
        eager_output = wrapper(*inputs)
    eager_metrics = {
        "layer_output": metrics(eager_output[0], expected[0]),
        "present_key": metrics(eager_output[1], expected[1]),
        "present_value": metrics(eager_output[2], expected[2]),
    }
    for name, output_metrics in eager_metrics.items():
        if output_metrics["maximum_abs_error"] > 1.0e-6:
            raise RuntimeError(
                f"Layer {layer_index} eager {name} failed: {output_metrics}"
            )

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            inputs,
            strict=True,
            check_trace=True,
        )
        traced = torch.jit.freeze(traced.eval())

    output_dir = ARTIFACTS_DIR / f"decoder_layer{layer_index}_prefill_kv"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_stem = f"decoder_layer{layer_index}_prefill_kv"
    script_path = output_dir / f"{model_stem}.pt"
    traced.save(str(script_path))

    with torch.inference_mode():
        traced_output = traced(*inputs)
    traced_metrics = {
        "layer_output": metrics(traced_output[0], expected[0]),
        "present_key": metrics(traced_output[1], expected[1]),
        "present_value": metrics(traced_output[2], expected[2]),
    }
    for name, output_metrics in traced_metrics.items():
        if output_metrics["maximum_abs_error"] > 1.0e-6:
            raise RuntimeError(
                f"Layer {layer_index} traced {name} failed: {output_metrics}"
            )

    pnnx_command: list[str] | None = None
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    pnnx_log_path = DOCS_DIR / f"decoder_layer{layer_index}_prefill_kv_pnnx.txt"
    if not skip_pnnx:
        pnnx_command = [
            str(PNNX_PATH),
            str(script_path),
            f"inputshape={shape_argument(inputs)}",
            "inputshape2=" + ",".join((
                f"[1,{ALTERNATE_SEQUENCE_LENGTH},{HIDDEN_SIZE}]",
                f"[1,1,{ALTERNATE_SEQUENCE_LENGTH},{ALTERNATE_SEQUENCE_LENGTH}]",
                f"[4,1,{ALTERNATE_SEQUENCE_LENGTH},{HEAD_DIM}]",
                f"[4,1,{ALTERNATE_SEQUENCE_LENGTH},{HEAD_DIM}]",
            )),
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
        normalized_log = "\n".join(
            line.rstrip() for line in conversion.stdout.splitlines()
        ) + "\n"
        pnnx_log_path.write_text(normalized_log, encoding="utf-8")
        if conversion.returncode != 0:
            raise RuntimeError(
                f"Layer {layer_index} pnnx failed with status "
                f"{conversion.returncode}; see {pnnx_log_path}"
            )

    report = {
        "layer_index": layer_index,
        "model_revision": (
            PROJECT_DIR / "configs/model_revision.txt"
        ).read_text(encoding="utf-8").strip(),
        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,
        "sequence_length": SEQUENCE_LENGTH,
        "alternate_sequence_length": ALTERNATE_SEQUENCE_LENGTH,
        "dynamic_sequence_length": True,
        "torchscript_path": str(script_path),
        "torchscript_bytes": script_path.stat().st_size,
        "eager_metrics": eager_metrics,
        "torchscript_metrics": traced_metrics,
        "pnnx_command": pnnx_command,
        "pnnx_log": (
            str(pnnx_log_path)
            if pnnx_command is not None
            else None
        ),
        "elapsed_seconds": time.perf_counter() - layer_start,
    }
    report_path = DOCS_DIR / f"decoder_layer{layer_index}_prefill_kv.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    del traced_output, traced, eager_output, wrapper, fixed_wrapper
    gc.collect()
    print(
        f"layer={layer_index:02d} eager_max="
        f"{max(value['maximum_abs_error'] for value in eager_metrics.values()):.3e} "
        f"traced_max="
        f"{max(value['maximum_abs_error'] for value in traced_metrics.values()):.3e} "
        f"elapsed={report['elapsed_seconds']:.3f}s",
        flush=True,
    )
    return report


def main() -> None:
    args = parse_args()
    global MODEL_DIR, ARTIFACTS_DIR, DOCS_DIR, REFERENCE_ROOT, PNNX_PATH
    MODEL_DIR = args.model_dir.resolve()
    ARTIFACTS_DIR = args.artifacts_dir.resolve()
    DOCS_DIR = args.docs_dir.resolve()
    REFERENCE_ROOT = args.reference_dir.resolve()
    PNNX_PATH = args.pnnx.resolve()
    if args.layer_index is not None and not 0 <= args.layer_index < LAYER_COUNT:
        raise ValueError("layer-index must be in [0, 23]")

    logging.disable_progress_bar()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)
    base = load_module(BASE_EXPORT_PATH, "prefill_export_base")

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
    print(f"model_load_seconds={time.perf_counter() - load_start:.3f}")

    layer_indices = (
        [args.layer_index]
        if args.layer_index is not None
        else list(range(LAYER_COUNT))
    )
    for layer_index in layer_indices:
        export_layer(layer_index, model, base, args.skip_pnnx)
    print(f"exported_layers={len(layer_indices)}")


if __name__ == "__main__":
    main()
