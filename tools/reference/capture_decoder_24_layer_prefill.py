from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from transformers import HunYuanVLForConditionalGeneration
from transformers.utils import logging


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
BASE_CAPTURE_SCRIPT = (
    PROJECT_DIR / "tools/reference/capture_decoder_layer0_decode_reference.py"
)
REPORT_PATH = PROJECT_DIR / "docs/decoder_24_layer_prefill_reference.json"

TORCH_THREADS = 9
LAYER_COUNT = 24
SEQUENCE_LENGTH = 313
HIDDEN_SIZE = 1024
KV_HEADS = 8
HEAD_DIM = 128
ROPE_COMPONENTS = 4
EXPECTED_FIRST_TOKEN = 93892


def load_module(path: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load Python module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = first_tensor(item)
            if result is not None:
                return result
    if isinstance(value, dict):
        for item in value.values():
            result = first_tensor(item)
            if result is not None:
                return result
    return None


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


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"Shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    return float((actual.float() - expected.float()).abs().max().item())


def save_tensor(
    reference_name: str,
    tensor_name: str,
    value: torch.Tensor,
) -> dict[str, Any]:
    detached = value.detach().cpu().contiguous()
    npy_dir = PROJECT_DIR / "reference/smoke_en_cpu_fp32" / reference_name
    raw_dir = PROJECT_DIR / "artifacts" / reference_name / "reference"
    npy_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    npy_path = npy_dir / f"{tensor_name}.npy"
    raw_path = raw_dir / f"{tensor_name}_f32.bin"
    array = detached.float().numpy()
    np.save(npy_path, array)
    array.tofile(raw_path)

    report = tensor_summary(detached)
    report.update(
        {
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
            "raw_bytes": raw_path.stat().st_size,
        }
    )
    return report


def load_reference(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.from_numpy(np.load(path)).float()


def main() -> None:
    capture_start = time.perf_counter()
    logging.disable_progress_bar()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    base = load_module(BASE_CAPTURE_SCRIPT, "prefill_capture_base")
    contract = base.load_contract_module()
    model_inputs = {
        name: contract.load_tensor(name)
        for name in (
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        )
    }

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
    decoder_layers = model.model.language_model.layers
    if len(decoder_layers) != LAYER_COUNT:
        raise RuntimeError(f"Expected {LAYER_COUNT} decoder layers")

    layer_inputs: list[torch.Tensor | None] = [None] * LAYER_COUNT
    layer_outputs: list[torch.Tensor | None] = [None] * LAYER_COUNT
    shared_attention_mask: torch.Tensor | None = None
    shared_rope_cos: torch.Tensor | None = None
    shared_rope_sin: torch.Tensor | None = None
    position_ids_are_none: list[bool] = []
    shared_input_errors = {
        "attention_mask": 0.0,
        "rope_cos": 0.0,
        "rope_sin": 0.0,
    }
    tail_captures: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def make_pre_hook(layer_index: int) -> Callable[..., None]:
        def hook(
            module: torch.nn.Module,
            hook_args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            nonlocal shared_attention_mask, shared_rope_cos, shared_rope_sin
            arguments = base.bind_forward_arguments(module, hook_args, kwargs)
            hidden = arguments.get("hidden_states")
            mask = arguments.get("attention_mask")
            position_embeddings = arguments.get("position_embeddings")
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError(f"Layer {layer_index} hidden input missing")
            if not isinstance(mask, torch.Tensor):
                raise RuntimeError(f"Layer {layer_index} attention mask missing")
            if not (
                isinstance(position_embeddings, tuple)
                and len(position_embeddings) == 2
                and all(isinstance(item, torch.Tensor) for item in position_embeddings)
            ):
                raise RuntimeError(f"Layer {layer_index} RoPE input missing")

            rope_cos, rope_sin = position_embeddings
            layer_inputs[layer_index] = hidden.detach().clone()
            position_ids_are_none.append(arguments.get("position_ids") is None)

            if layer_index == 0:
                shared_attention_mask = mask.detach().clone()
                shared_rope_cos = rope_cos.detach().clone()
                shared_rope_sin = rope_sin.detach().clone()
            else:
                assert shared_attention_mask is not None
                assert shared_rope_cos is not None
                assert shared_rope_sin is not None
                shared_input_errors["attention_mask"] = max(
                    shared_input_errors["attention_mask"],
                    max_abs_error(mask, shared_attention_mask),
                )
                shared_input_errors["rope_cos"] = max(
                    shared_input_errors["rope_cos"],
                    max_abs_error(rope_cos, shared_rope_cos),
                )
                shared_input_errors["rope_sin"] = max(
                    shared_input_errors["rope_sin"],
                    max_abs_error(rope_sin, shared_rope_sin),
                )

        return hook

    def make_post_hook(layer_index: int) -> Callable[..., None]:
        def hook(
            module: torch.nn.Module,
            hook_args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, hook_args, kwargs
            value = first_tensor(output)
            if value is None:
                raise RuntimeError(f"Layer {layer_index} output missing")
            layer_outputs[layer_index] = value.detach().clone()

        return hook

    for layer_index, layer in enumerate(decoder_layers):
        handles.append(
            layer.register_forward_pre_hook(
                make_pre_hook(layer_index), with_kwargs=True
            )
        )
        handles.append(
            layer.register_forward_hook(
                make_post_hook(layer_index), with_kwargs=True
            )
        )

    def norm_hook(
        module: torch.nn.Module,
        module_inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module
        if module_inputs and isinstance(module_inputs[0], torch.Tensor):
            tail_captures["final_norm_input"] = module_inputs[0].detach().clone()
        if isinstance(output, torch.Tensor):
            tail_captures["final_norm_output"] = output.detach().clone()

    def lm_head_hook(
        module: torch.nn.Module,
        module_inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module
        if module_inputs and isinstance(module_inputs[0], torch.Tensor):
            tail_captures["lm_head_input"] = module_inputs[0].detach().clone()
        if isinstance(output, torch.Tensor):
            tail_captures["prefill_logits"] = output.detach().clone()

    handles.append(model.model.language_model.norm.register_forward_hook(norm_hook))
    handles.append(model.lm_head.register_forward_hook(lm_head_hook))

    forward_start = time.perf_counter()
    with torch.inference_mode():
        output = model(
            **model_inputs,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
    forward_seconds = time.perf_counter() - forward_start
    for handle in handles:
        handle.remove()

    if any(value is None for value in layer_inputs + layer_outputs):
        raise RuntimeError("One or more decoder layer tensors were not captured")
    if shared_attention_mask is None or shared_rope_cos is None or shared_rope_sin is None:
        raise RuntimeError("Shared prefill inputs were not captured")
    if not all(position_ids_are_none) or len(position_ids_are_none) != LAYER_COUNT:
        raise RuntimeError("Expected position_ids=None for all decoder layers")
    if any(error != 0.0 for error in shared_input_errors.values()):
        raise RuntimeError(f"Per-layer mask/RoPE mismatch: {shared_input_errors}")

    legacy_cache = contract.cache_to_legacy(output.past_key_values)
    if len(legacy_cache) != LAYER_COUNT:
        raise RuntimeError(f"Expected {LAYER_COUNT} cache layers")
    actual_first_token = int(output.logits[0, -1].argmax(dim=-1).item())
    if actual_first_token != EXPECTED_FIRST_TOKEN:
        raise RuntimeError(
            f"Prefill token {actual_first_token} != {EXPECTED_FIRST_TOKEN}"
        )

    layer_reports: list[dict[str, Any]] = []
    maximum_boundary_error = 0.0
    maximum_existing_cache_error = 0.0
    for layer_index in range(LAYER_COUNT):
        hidden = layer_inputs[layer_index]
        layer_output = layer_outputs[layer_index]
        assert hidden is not None and layer_output is not None
        present_key, present_value = legacy_cache[layer_index]
        expected_shapes = (
            (1, SEQUENCE_LENGTH, HIDDEN_SIZE),
            (1, SEQUENCE_LENGTH, HIDDEN_SIZE),
            (1, KV_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
            (1, KV_HEADS, SEQUENCE_LENGTH, HEAD_DIM),
        )
        for value, expected_shape in zip(
            (hidden, layer_output, present_key, present_value), expected_shapes
        ):
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"Layer {layer_index} shape {tuple(value.shape)} != {expected_shape}"
                )

        boundary_error = 0.0
        if layer_index > 0:
            previous_output = layer_outputs[layer_index - 1]
            assert previous_output is not None
            boundary_error = max_abs_error(hidden, previous_output)
            maximum_boundary_error = max(maximum_boundary_error, boundary_error)

        decode_reference = (
            PROJECT_DIR
            / "reference/smoke_en_cpu_fp32"
            / f"decoder_layer{layer_index}_decode"
        )
        existing_key = load_reference(decode_reference / "past_key.npy")
        existing_value = load_reference(decode_reference / "past_value.npy")
        key_error = max_abs_error(present_key, existing_key)
        value_error = max_abs_error(present_value, existing_value)
        maximum_existing_cache_error = max(
            maximum_existing_cache_error, key_error, value_error
        )

        reference_name = f"decoder_layer{layer_index}_prefill_kv"
        prefix = f"layer{layer_index}"
        tensors = {
            "hidden_states": save_tensor(
                reference_name, f"{prefix}_hidden_states", hidden
            ),
            "layer_output": save_tensor(
                reference_name, f"{prefix}_output", layer_output
            ),
            "present_key": save_tensor(reference_name, "present_key", present_key),
            "present_value": save_tensor(
                reference_name, "present_value", present_value
            ),
        }
        if layer_index == 0:
            tensors.update(
                {
                    "attention_mask": save_tensor(
                        reference_name,
                        "layer0_attention_mask",
                        shared_attention_mask,
                    ),
                    "rope_cos": save_tensor(
                        reference_name,
                        "layer0_position_embeddings_0",
                        shared_rope_cos,
                    ),
                    "rope_sin": save_tensor(
                        reference_name,
                        "layer0_position_embeddings_1",
                        shared_rope_sin,
                    ),
                }
            )
        layer_reports.append(
            {
                "layer_index": layer_index,
                "boundary_max_abs_error": boundary_error,
                "decode_reference_key_max_abs_error": key_error,
                "decode_reference_value_max_abs_error": value_error,
                "tensors": tensors,
            }
        )

    layer0_contract_dir = (
        PROJECT_DIR / "reference/smoke_en_cpu_fp32/decoder_layer0_prefill"
    )
    layer0_contract_errors = {
        "hidden_states": max_abs_error(
            layer_inputs[0], load_reference(layer0_contract_dir / "layer_input.npy")
        ),
        "attention_mask": max_abs_error(
            shared_attention_mask,
            load_reference(layer0_contract_dir / "attention_mask.npy"),
        ),
        "rope_cos": max_abs_error(
            shared_rope_cos, load_reference(layer0_contract_dir / "rope_cos.npy")
        ),
        "rope_sin": max_abs_error(
            shared_rope_sin, load_reference(layer0_contract_dir / "rope_sin.npy")
        ),
    }
    if any(error != 0.0 for error in layer0_contract_errors.values()):
        raise RuntimeError(f"Layer 0 contract changed: {layer0_contract_errors}")
    if maximum_boundary_error != 0.0 or maximum_existing_cache_error != 0.0:
        raise RuntimeError("Captured prefill boundaries or KV caches changed")

    final_norm_input = tail_captures["final_norm_input"][:, -1:, :]
    final_norm_output = tail_captures["lm_head_input"]
    logits = output.logits
    tail_tensors = {
        "final_norm_input": save_tensor(
            "prefill_tail", "final_norm_input", final_norm_input
        ),
        "final_norm_output": save_tensor(
            "prefill_tail", "final_norm_output", final_norm_output
        ),
        "prefill_logits": save_tensor("prefill_tail", "prefill_logits", logits),
    }

    mask_numeric = shared_attention_mask.float()
    report = {
        "model_revision": (
            PROJECT_DIR / "configs/model_revision.txt"
        ).read_text(encoding="utf-8").strip(),
        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": model_load_seconds,
        "prefill_forward_seconds": forward_seconds,
        "total_capture_seconds": time.perf_counter() - capture_start,
        "sequence_length": SEQUENCE_LENGTH,
        "layer_count": LAYER_COUNT,
        "position_ids_are_none": True,
        "shared_input_max_abs_errors": shared_input_errors,
        "layer0_contract_max_abs_errors": layer0_contract_errors,
        "maximum_layer_boundary_error": maximum_boundary_error,
        "maximum_existing_cache_error": maximum_existing_cache_error,
        "attention_mask": {
            **tensor_summary(shared_attention_mask),
            "zero_count": int((mask_numeric == 0).sum().item()),
            "negative_count": int((mask_numeric < 0).sum().item()),
        },
        "rope_cos": tensor_summary(shared_rope_cos),
        "rope_sin": tensor_summary(shared_rope_sin),
        "expected_first_token": EXPECTED_FIRST_TOKEN,
        "actual_first_token": actual_first_token,
        "tail_tensors": tail_tensors,
        "layers": layer_reports,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"model_load_seconds={model_load_seconds:.3f}")
    print(f"prefill_forward_seconds={forward_seconds:.3f}")
    print(f"layers={len(layer_reports)}/{LAYER_COUNT}")
    print(f"maximum_layer_boundary_error={maximum_boundary_error:.10e}")
    print(f"maximum_existing_cache_error={maximum_existing_cache_error:.10e}")
    print(f"first_token={actual_first_token}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
