from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import HunYuanVLForConditionalGeneration
from transformers.utils import logging


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
PNNX_PATH = Path.home() / "work/hunyuanocr/.venv-pnnx/bin/pnnx"

TORCH_THREADS = 9
LAYER_COUNT = 27
PATCH_COUNT = 1100
PATCH_VECTOR_SIZE = 768
PATCH_SIZE = 16
CHANNELS = 3
GRID_H = 22
GRID_W = 50
VISION_HIDDEN_SIZE = 1152
INTERMEDIATE_SIZE = 4304
ATTENTION_HEADS = 16
HEAD_DIM = 72
MERGE_SIZE = 2
MERGED_H = 11
MERGED_W = 25
MERGED_TOKEN_COUNT = 288
TEXT_HIDDEN_SIZE = 1024
EXPORT_MAXIMUM_ABS_ERROR = 1.0e-2
EXPORT_MEAN_ABS_ERROR = 2.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the fixed-grid HunyuanOCR patch embedding, all vision "
            "blocks, and patch merger."
        )
    )
    parser.add_argument(
        "--component",
        choices=("all", "patch", "block", "merger"),
        default="all",
    )
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--skip-pnnx", action="store_true")
    return parser.parse_args()


def load_f32(reference_name: str, tensor_name: str) -> torch.Tensor:
    path = (
        PROJECT_DIR
        / "reference/smoke_en_cpu_fp32"
        / reference_name
        / f"{tensor_name}.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.from_numpy(np.load(path)).float().contiguous()


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


class PatchEmbeddingWrapper(nn.Module):
    def __init__(self, embeddings: nn.Module) -> None:
        super().__init__()
        self.patch_embedding = embeddings.patch_embedding
        example = torch.empty(
            1, PATCH_COUNT, VISION_HIDDEN_SIZE, dtype=torch.float32
        )
        with torch.no_grad():
            position = embeddings.interpolate_pos_encoding(
                example, GRID_H, GRID_W
            )
        self.register_buffer(
            "position_embedding", position.detach().float().contiguous()
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixels = pixel_values.reshape(
            PATCH_COUNT, CHANNELS, PATCH_SIZE, PATCH_SIZE
        )
        patch_embeddings = self.patch_embedding(pixels)
        patch_embeddings = (
            patch_embeddings.flatten(-2)
            .squeeze(-1)
            .reshape(1, PATCH_COUNT, VISION_HIDDEN_SIZE)
        )
        return patch_embeddings + self.position_embedding


class VisionBlockWrapper(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer_norm1 = layer.layer_norm1
        self.q_proj = layer.self_attn.q_proj
        self.k_proj = layer.self_attn.k_proj
        self.v_proj = layer.self_attn.v_proj
        self.o_proj = layer.self_attn.o_proj
        self.scaling = float(layer.self_attn.scaling)
        self.layer_norm2 = layer.layer_norm2
        self.fc1 = layer.mlp.fc1
        self.activation_fn = layer.mlp.activation_fn
        self.fc2 = layer.mlp.fc2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        normalized = self.layer_norm1(hidden_states)
        query = self.q_proj(normalized).reshape(
            PATCH_COUNT, ATTENTION_HEADS, HEAD_DIM
        ).transpose(0, 1)
        key = self.k_proj(normalized).reshape(
            PATCH_COUNT, ATTENTION_HEADS, HEAD_DIM
        ).transpose(0, 1)
        value = self.v_proj(normalized).reshape(
            PATCH_COUNT, ATTENTION_HEADS, HEAD_DIM
        ).transpose(0, 1)
        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attention_output = (
            attention_output.transpose(0, 1)
            .contiguous()
            .reshape(PATCH_COUNT, VISION_HIDDEN_SIZE)
        )
        hidden_states = residual + self.o_proj(attention_output)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        return residual + hidden_states


class PatchMergerWrapper(nn.Module):
    def __init__(self, merger: nn.Module) -> None:
        super().__init__()
        self.before_rms = merger.before_rms
        self.proj_conv = merger.proj_conv
        self.proj_act = merger.proj_act
        self.proj_out = merger.proj_out
        self.mlp = merger.mlp
        self.image_newline = merger.image_newline
        self.image_begin = merger.image_begin
        self.image_end = merger.image_end
        self.after_rms = merger.after_rms

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.before_rms(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            1, VISION_HIDDEN_SIZE, GRID_H, GRID_W
        )
        hidden_states = self.proj_conv(hidden_states)
        hidden_states = self.proj_act(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        newline = self.image_newline.reshape(1, -1, 1, 1).expand(
            1, VISION_HIDDEN_SIZE * 4, MERGED_H, 1
        )
        hidden_states = torch.cat((hidden_states, newline), dim=-1)
        hidden_states = hidden_states.reshape(
            1, VISION_HIDDEN_SIZE * 4, MERGED_H * (MERGED_W + 1)
        ).permute(0, 2, 1)
        hidden_states = self.mlp(hidden_states)
        begin = self.image_begin.reshape(1, 1, TEXT_HIDDEN_SIZE)
        end = self.image_end.reshape(1, 1, TEXT_HIDDEN_SIZE)
        hidden_states = torch.cat((begin, hidden_states, end), dim=1)
        return self.after_rms(hidden_states)


def shape_argument(values: tuple[torch.Tensor, ...]) -> str:
    return ",".join(
        "[" + ",".join(str(dimension) for dimension in value.shape) + "]"
        for value in values
    )


def normalize_vision_sdpa_layout(param_path: Path) -> None:
    lines = param_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3 or lines[0] != "7767517":
        raise RuntimeError(f"Unexpected ncnn param header: {param_path}")

    counts = lines[1].split()
    if len(counts) != 2:
        raise RuntimeError(f"Unexpected ncnn graph counts: {param_path}")

    parsed: list[list[str]] = [line.split() for line in lines[2:]]
    sdpa_indices = [
        index for index, tokens in enumerate(parsed)
        if tokens and tokens[0] == "SDPA"
    ]
    if len(sdpa_indices) != 1:
        raise RuntimeError(
            f"Expected exactly one SDPA in {param_path}, got "
            f"{len(sdpa_indices)}"
        )

    sdpa_index = sdpa_indices[0]
    sdpa = parsed[sdpa_index]
    bottom_count = int(sdpa[2])
    if bottom_count != 3 or int(sdpa[3]) != 1:
        raise RuntimeError(f"Unexpected SDPA signature: {' '.join(sdpa)}")

    inserted = 0
    sdpa_inputs = list(sdpa[4:7])
    for input_offset, input_blob in enumerate(sdpa_inputs):
        producer_index = -1
        for index, tokens in enumerate(parsed):
            if len(tokens) < 5:
                continue
            producer_bottoms = int(tokens[2])
            producer_tops = int(tokens[3])
            top_start = 4 + producer_bottoms
            top_end = top_start + producer_tops
            if input_blob in tokens[top_start:top_end]:
                producer_index = index
                break
        if producer_index < 0 or parsed[producer_index][0] != "Reshape":
            raise RuntimeError(
                f"SDPA input {input_blob} is not produced by Reshape"
            )

        producer = parsed[producer_index]
        parameters = {
            token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in producer
            if "=" in token
        }
        expected_shape = {
            "0": str(HEAD_DIM),
            "1": str(ATTENTION_HEADS),
            "2": str(PATCH_COUNT),
        }
        if any(parameters.get(key) != value
               for key, value in expected_shape.items()):
            raise RuntimeError(
                f"Unexpected SDPA reshape: {' '.join(producer)}"
            )

        producer[:] = [
            token for token in producer
            if not token.startswith("12=") and not token.startswith("13=")
        ]
        permuted_blob = f"{input_blob}_vision_heads"
        permute = [
            "Permute",
            f"vision_head_layout_{input_offset}",
            "1",
            "1",
            input_blob,
            permuted_blob,
            "0=2",
        ]
        parsed.insert(producer_index + 1, permute)
        if producer_index < sdpa_index:
            sdpa_index += 1
        inserted += 1
        parsed[sdpa_index][4 + input_offset] = permuted_blob

    sdpa_output = parsed[sdpa_index][7]
    output_permute = next(
        (
            tokens for tokens in parsed
            if tokens[0] == "Permute" and tokens[4] == sdpa_output
        ),
        None,
    )
    if output_permute is None:
        raise RuntimeError("SDPA output is not consumed by Permute")
    permute_output = output_permute[5]
    output_reshape = next(
        (
            tokens for tokens in parsed
            if tokens[0] == "Reshape" and tokens[4] == permute_output
        ),
        None,
    )
    if output_reshape is None:
        raise RuntimeError("SDPA output Permute is not consumed by Reshape")
    output_parameters = {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in output_reshape
        if "=" in token
    }
    if (output_parameters.get("0") != str(VISION_HIDDEN_SIZE)
            or output_parameters.get("1") != str(PATCH_COUNT)):
        raise RuntimeError(
            f"Unexpected SDPA output reshape: {' '.join(output_reshape)}"
        )
    output_reshape[:] = [
        token for token in output_reshape
        if not token.startswith("12=") and not token.startswith("13=")
    ]

    counts[0] = str(int(counts[0]) + inserted)
    counts[1] = str(int(counts[1]) + inserted)
    output_lines = [lines[0], " ".join(counts)] + [
        " ".join(tokens) for tokens in parsed
    ]
    param_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def export_component(
    name: str,
    wrapper: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    expected: torch.Tensor,
    skip_pnnx: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    component_start = time.perf_counter()
    wrapper = wrapper.eval()
    with torch.inference_mode():
        eager_output = wrapper(*inputs)
    eager_metrics = metrics(eager_output, expected)
    if (eager_metrics["maximum_abs_error"] > EXPORT_MAXIMUM_ABS_ERROR
            or eager_metrics["mean_abs_error"] > EXPORT_MEAN_ABS_ERROR):
        raise RuntimeError(f"{name} eager parity failed: {eager_metrics}")

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            inputs,
            strict=True,
            check_trace=True,
        )
        traced = torch.jit.freeze(traced.eval())

    output_dir = PROJECT_DIR / "artifacts" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{name}.pt"
    traced.save(str(script_path))

    with torch.inference_mode():
        traced_output = traced(*inputs)
    traced_metrics = metrics(traced_output, expected)
    if (traced_metrics["maximum_abs_error"] > EXPORT_MAXIMUM_ABS_ERROR
            or traced_metrics["mean_abs_error"] > EXPORT_MEAN_ABS_ERROR):
        raise RuntimeError(f"{name} TorchScript parity failed: {traced_metrics}")

    pnnx_command: list[str] | None = None
    pnnx_log_path = PROJECT_DIR / "docs" / f"{name}_pnnx.txt"
    if not skip_pnnx:
        pnnx_command = [
            str(PNNX_PATH),
            str(script_path),
            f"inputshape={shape_argument(inputs)}",
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
                f"{name} pnnx failed with status {conversion.returncode}; "
                f"see {pnnx_log_path}"
            )
        if name.startswith("vision_block"):
            normalize_vision_sdpa_layout(
                output_dir / f"{name}.ncnn.param"
            )

    report = {
        "component": name,
        "model_revision": (
            PROJECT_DIR / "configs/model_revision.txt"
        ).read_text(encoding="utf-8").strip(),
        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,
        "inputs": [list(value.shape) for value in inputs],
        "output": list(expected.shape),
        "eager_metrics": eager_metrics,
        "torchscript_metrics": traced_metrics,
        "torchscript_path": script_path.relative_to(PROJECT_DIR).as_posix(),
        "torchscript_bytes": script_path.stat().st_size,
        "pnnx_command": pnnx_command,
        "pnnx_log": (
            pnnx_log_path.relative_to(PROJECT_DIR).as_posix()
            if pnnx_command is not None
            else None
        ),
        "elapsed_seconds": time.perf_counter() - component_start,
        **metadata,
    }
    report_path = PROJECT_DIR / "docs" / f"{name}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"component={name} eager_max={eager_metrics['maximum_abs_error']:.3e} "
        f"traced_max={traced_metrics['maximum_abs_error']:.3e} "
        f"elapsed={report['elapsed_seconds']:.3f}s",
        flush=True,
    )
    del eager_output, traced_output, traced, wrapper
    gc.collect()
    return report


def main() -> None:
    args = parse_args()
    if args.layer_index is not None and not 0 <= args.layer_index < LAYER_COUNT:
        raise ValueError("layer-index must be in [0, 26]")
    if args.component != "block" and args.layer_index is not None:
        raise ValueError("layer-index is only valid for --component block")

    logging.disable_progress_bar()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

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
    vision_tower = model.model.vision_tower
    print(f"model_load_seconds={time.perf_counter() - load_start:.3f}")
    exported = 0

    if args.component in ("all", "patch"):
        pixel_values = load_f32("vision_patch_embed", "pixel_values")
        expected = load_f32("vision_patch_embed", "expected_output")
        export_component(
            "vision_patch_embed",
            PatchEmbeddingWrapper(vision_tower.embeddings),
            (pixel_values,),
            expected,
            args.skip_pnnx,
            {
                "grid_thw": [[1, GRID_H, GRID_W]],
                "position_encoding": "precomputed bilinear 22x50 patch grid",
            },
        )
        exported += 1

    if args.component in ("all", "block"):
        layer_indices = (
            [args.layer_index]
            if args.layer_index is not None
            else list(range(LAYER_COUNT))
        )
        for layer_index in layer_indices:
            name = f"vision_block{layer_index}"
            hidden = load_f32(name, "hidden_states").squeeze(0)
            expected = load_f32(name, "expected_output").squeeze(0)
            export_component(
                name,
                VisionBlockWrapper(vision_tower.layers[layer_index]),
                (hidden,),
                expected,
                args.skip_pnnx,
                {
                    "layer_index": layer_index,
                    "attention_mask": None,
                    "cu_seqlens": [0, PATCH_COUNT],
                    "rotary_position_embedding": False,
                },
            )
            exported += 1

    if args.component in ("all", "merger"):
        hidden = load_f32("vision_patch_merger", "hidden_states")
        expected = load_f32("vision_patch_merger", "expected_output")
        export_component(
            "vision_patch_merger",
            PatchMergerWrapper(vision_tower.patch_merger),
            (hidden,),
            expected,
            args.skip_pnnx,
            {
                "grid_thw": [[1, GRID_H, GRID_W]],
                "spatial_merge_size": MERGE_SIZE,
                "merged_spatial_shape": [MERGED_H, MERGED_W],
                "output_token_count": MERGED_TOKEN_COUNT,
            },
        )
        exported += 1

    print(f"exported_components={exported}")


if __name__ == "__main__":
    main()
