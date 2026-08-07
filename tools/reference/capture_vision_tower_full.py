from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from transformers import HunYuanVLForConditionalGeneration
from transformers.utils import logging


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
MODEL_DIR = Path(os.environ.get("HUNYUANOCR_MODEL_DIR", str(MODEL_DIR)))
REFERENCE_ROOT = Path(os.environ.get(
    "HUNYUANOCR_REFERENCE_DIR",
    str(PROJECT_DIR / "reference/smoke_en_cpu_fp32"),
))
ARTIFACTS_DIR = Path(os.environ.get(
    "HUNYUANOCR_ARTIFACTS_DIR",
    str(PROJECT_DIR / "artifacts"),
))
DOCS_DIR = Path(os.environ.get("HUNYUANOCR_DOCS_DIR", str(PROJECT_DIR / "docs")))
CONTRACT_SCRIPT = (
    PROJECT_DIR / "tools/inspect/inspect_decoder_layer0_decode_contract.py"
)
REPORT_PATH = DOCS_DIR / "vision_tower_full_reference.json"

TORCH_THREADS = 9
LAYER_COUNT = 27
PATCH_COUNT = 1100
PATCH_VECTOR_SIZE = 768
VISION_HIDDEN_SIZE = 1152
TEXT_HIDDEN_SIZE = 1024
GRID_T = 1
GRID_H = 22
GRID_W = 50
MERGED_TOKEN_COUNT = 288


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load Python module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().cpu()
    numeric = detached.float()
    finite = numeric[torch.isfinite(numeric)]
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "minimum_finite": float(finite.min().item()),
        "maximum_finite": float(finite.max().item()),
        "mean_finite": float(finite.mean().item()),
        "nan_count": int(torch.isnan(numeric).sum().item()),
        "positive_inf_count": int(torch.isposinf(numeric).sum().item()),
        "negative_inf_count": int(torch.isneginf(numeric).sum().item()),
    }


def dtype_tag(value: torch.Tensor) -> str:
    return {
        torch.float32: "f32",
        torch.int64: "i64",
        torch.int32: "i32",
    }.get(value.dtype, str(value.dtype).replace("torch.", ""))


def save_tensor(
    reference_name: str,
    tensor_name: str,
    value: torch.Tensor,
) -> dict[str, Any]:
    detached = value.detach().cpu().contiguous()
    npy_dir = REFERENCE_ROOT / reference_name
    raw_dir = ARTIFACTS_DIR / reference_name / "reference"
    npy_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    npy_path = npy_dir / f"{tensor_name}.npy"
    raw_path = raw_dir / f"{tensor_name}_{dtype_tag(detached)}.bin"
    array = detached.numpy()
    np.save(npy_path, array)
    array.tofile(raw_path)

    report = tensor_summary(detached)
    report.update(
        {
            "npy_path": str(npy_path),
            "raw_path": str(raw_path),
            "raw_bytes": raw_path.stat().st_size,
        }
    )
    return report


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"Shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    return float((actual.float() - expected.float()).abs().max().item())


def load_f32(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.from_numpy(np.load(path)).float()


def main() -> None:
    capture_start = time.perf_counter()
    logging.disable_progress_bar()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    contract = load_module(CONTRACT_SCRIPT, "vision_capture_contract")
    pixel_values = contract.load_tensor("pixel_values").float().contiguous()
    image_grid_thw = contract.load_tensor("image_grid_thw").long().contiguous()
    if tuple(pixel_values.shape) != (PATCH_COUNT, PATCH_VECTOR_SIZE):
        raise RuntimeError(f"Unexpected pixel_values shape: {pixel_values.shape}")
    if image_grid_thw.tolist() != [[GRID_T, GRID_H, GRID_W]]:
        raise RuntimeError(f"Unexpected image_grid_thw: {image_grid_thw.tolist()}")

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
    model_load_seconds = time.perf_counter() - load_start
    vision_tower = model.model.vision_tower
    if len(vision_tower.layers) != LAYER_COUNT:
        raise RuntimeError(f"Expected {LAYER_COUNT} vision blocks")

    captures: dict[str, torch.Tensor] = {}
    block_inputs: list[torch.Tensor | None] = [None] * LAYER_COUNT
    block_outputs: list[torch.Tensor | None] = [None] * LAYER_COUNT
    cu_seqlens_values: list[list[int]] = []
    attention_masks_are_none: list[bool] = []
    merger_size: tuple[int, int] | None = None
    handles: list[Any] = []

    def patch_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        del module, args, kwargs
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Patch embedding output is not a tensor")
        captures["patch_output"] = output.detach().clone()

    handles.append(
        vision_tower.embeddings.register_forward_hook(
            patch_hook, with_kwargs=True
        )
    )

    def make_block_pre_hook(layer_index: int) -> Callable[..., None]:
        def hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            del module
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            cu_seqlens = kwargs.get("cu_seqlens")
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError(f"Vision block {layer_index} input missing")
            if not isinstance(cu_seqlens, torch.Tensor):
                raise RuntimeError(f"Vision block {layer_index} cu_seqlens missing")
            block_inputs[layer_index] = hidden.detach().clone()
            cu_seqlens_values.append([int(item) for item in cu_seqlens.tolist()])
            attention_masks_are_none.append(kwargs.get("attention_mask") is None)

        return hook

    def make_block_post_hook(layer_index: int) -> Callable[..., None]:
        def hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, args, kwargs
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"Vision block {layer_index} output missing")
            block_outputs[layer_index] = output.detach().clone()

        return hook

    for layer_index, layer in enumerate(vision_tower.layers):
        handles.append(
            layer.register_forward_pre_hook(
                make_block_pre_hook(layer_index), with_kwargs=True
            )
        )
        handles.append(
            layer.register_forward_hook(
                make_block_post_hook(layer_index), with_kwargs=True
            )
        )

    def merger_pre_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        nonlocal merger_size
        del module
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        size = kwargs.get("size", args[1] if len(args) > 1 else None)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Patch merger input missing")
        if not isinstance(size, tuple) or len(size) != 2:
            raise RuntimeError(f"Patch merger size missing: {size}")
        captures["merger_input"] = hidden.detach().clone()
        merger_size = (int(size[0]), int(size[1]))

    def merger_post_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        del module, args, kwargs
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Patch merger output missing")
        captures["merger_output"] = output.detach().clone()

    handles.append(
        vision_tower.patch_merger.register_forward_pre_hook(
            merger_pre_hook, with_kwargs=True
        )
    )
    handles.append(
        vision_tower.patch_merger.register_forward_hook(
            merger_post_hook, with_kwargs=True
        )
    )

    forward_start = time.perf_counter()
    with torch.inference_mode():
        output = vision_tower(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
        )
    forward_seconds = time.perf_counter() - forward_start
    for handle in handles:
        handle.remove()

    if any(value is None for value in block_inputs + block_outputs):
        raise RuntimeError("One or more vision block boundaries were not captured")
    if merger_size != (GRID_H, GRID_W):
        raise RuntimeError(f"Unexpected merger size: {merger_size}")
    if not all(attention_masks_are_none):
        raise RuntimeError("Vision attention unexpectedly received a mask")
    if cu_seqlens_values != [[0, PATCH_COUNT]] * LAYER_COUNT:
        raise RuntimeError(f"Unexpected cu_seqlens: {cu_seqlens_values}")

    patch_output = captures["patch_output"]
    merger_input = captures["merger_input"]
    merger_output = captures["merger_output"]
    if tuple(patch_output.shape) != (1, PATCH_COUNT, VISION_HIDDEN_SIZE):
        raise RuntimeError(f"Unexpected patch output: {patch_output.shape}")
    if tuple(merger_output.shape) != (1, MERGED_TOKEN_COUNT, TEXT_HIDDEN_SIZE):
        raise RuntimeError(f"Unexpected merger output: {merger_output.shape}")

    split_dir = REFERENCE_ROOT / "split_contract"
    expected_last_hidden = load_f32(split_dir / "vision_last_hidden_state.npy")
    expected_pooler = load_f32(split_dir / "vision_pooler_output.npy")
    last_hidden_error = max_abs_error(output.last_hidden_state, expected_last_hidden)
    pooler_error = max_abs_error(output.pooler_output, expected_pooler)
    if last_hidden_error != 0.0 or pooler_error != 0.0:
        raise RuntimeError(
            f"Vision output changed: hidden={last_hidden_error}, pooler={pooler_error}"
        )

    patch_tensors = {
        "pixel_values": save_tensor(
            "vision_patch_embed", "pixel_values", pixel_values
        ),
        "image_grid_thw": save_tensor(
            "vision_patch_embed", "image_grid_thw", image_grid_thw
        ),
        "expected_output": save_tensor(
            "vision_patch_embed", "expected_output", patch_output
        ),
    }

    maximum_boundary_error = max_abs_error(patch_output, block_inputs[0])
    block_reports: list[dict[str, Any]] = []
    for layer_index in range(LAYER_COUNT):
        hidden = block_inputs[layer_index]
        block_output = block_outputs[layer_index]
        assert hidden is not None and block_output is not None
        boundary_error = 0.0
        if layer_index > 0:
            previous = block_outputs[layer_index - 1]
            assert previous is not None
            boundary_error = max_abs_error(hidden, previous)
        maximum_boundary_error = max(maximum_boundary_error, boundary_error)
        reference_name = f"vision_block{layer_index}"
        block_reports.append(
            {
                "layer_index": layer_index,
                "boundary_max_abs_error": boundary_error,
                "tensors": {
                    "hidden_states": save_tensor(
                        reference_name, "hidden_states", hidden
                    ),
                    "expected_output": save_tensor(
                        reference_name, "expected_output", block_output
                    ),
                },
            }
        )

    merger_boundary_error = max_abs_error(block_outputs[-1], merger_input)
    maximum_boundary_error = max(maximum_boundary_error, merger_boundary_error)
    merger_tensors = {
        "hidden_states": save_tensor(
            "vision_patch_merger", "hidden_states", merger_input
        ),
        "image_grid_thw": save_tensor(
            "vision_patch_merger", "image_grid_thw", image_grid_thw
        ),
        "expected_output": save_tensor(
            "vision_patch_merger", "expected_output", merger_output
        ),
    }
    if maximum_boundary_error != 0.0:
        raise RuntimeError(f"Vision boundary error: {maximum_boundary_error}")

    rotary_modules = [
        module.__class__.__name__
        for module in vision_tower.modules()
        if "rotary" in module.__class__.__name__.lower()
        or "rope" in module.__class__.__name__.lower()
    ]
    report = {
        "model_revision": (
            PROJECT_DIR / "configs/model_revision.txt"
        ).read_text(encoding="utf-8").strip(),
        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": model_load_seconds,
        "vision_forward_seconds": forward_seconds,
        "total_capture_seconds": time.perf_counter() - capture_start,
        "pixel_values_shape": list(pixel_values.shape),
        "image_grid_thw": image_grid_thw.tolist(),
        "patch_size": int(vision_tower.config.patch_size),
        "patch_count": PATCH_COUNT,
        "vision_hidden_size": VISION_HIDDEN_SIZE,
        "intermediate_size": int(vision_tower.config.intermediate_size),
        "attention_heads": int(vision_tower.config.attention_heads),
        "attention_head_dim": (
            VISION_HIDDEN_SIZE // int(vision_tower.config.attention_heads)
        ),
        "layer_count": LAYER_COUNT,
        "attention_mask_is_none": True,
        "cu_seqlens": [0, PATCH_COUNT],
        "rotary_module_count": len(rotary_modules),
        "rotary_modules": rotary_modules,
        "position_encoding": (
            "learned 128x128 patch grid bilinearly interpolated to 22x50; "
            "no vision RoPE"
        ),
        "spatial_merge_size": int(vision_tower.config.spatial_merge_size),
        "merged_spatial_shape": [GRID_H // 2, GRID_W // 2],
        "newline_token_count": GRID_H // 2,
        "begin_token_count": 1,
        "end_token_count": 1,
        "merged_token_count": MERGED_TOKEN_COUNT,
        "text_hidden_size": TEXT_HIDDEN_SIZE,
        "parameter_counts": {
            "patch_embedding": sum(
                parameter.numel()
                for parameter in vision_tower.embeddings.parameters()
            ),
            "vision_blocks": [
                sum(parameter.numel() for parameter in layer.parameters())
                for layer in vision_tower.layers
            ],
            "patch_merger": sum(
                parameter.numel()
                for parameter in vision_tower.patch_merger.parameters()
            ),
        },
        "maximum_layer_boundary_error": maximum_boundary_error,
        "split_contract_last_hidden_max_abs_error": last_hidden_error,
        "split_contract_pooler_max_abs_error": pooler_error,
        "patch_embedding": patch_tensors,
        "blocks": block_reports,
        "patch_merger": {
            "boundary_max_abs_error": merger_boundary_error,
            "tensors": merger_tensors,
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"model_load_seconds={model_load_seconds:.3f}")
    print(f"vision_forward_seconds={forward_seconds:.3f}")
    print(f"patch_output_shape={tuple(patch_output.shape)}")
    print(f"vision_blocks={len(block_reports)}/{LAYER_COUNT}")
    print(f"maximum_layer_boundary_error={maximum_boundary_error:.10e}")
    print(f"rotary_module_count={len(rotary_modules)}")
    print(f"merger_output_shape={tuple(merger_output.shape)}")
    print(f"split_contract_pooler_error={pooler_error:.10e}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
