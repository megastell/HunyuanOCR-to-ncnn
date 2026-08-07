from __future__ import annotations

import argparse
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
BASE_CAPTURE_SCRIPT = (
    PROJECT_DIR / "tools/reference/capture_decoder_layer0_decode_reference.py"
)
GENERATED_TOKENS_PATH = REFERENCE_ROOT / "generated_token_ids.json"

TORCH_THREADS = 9
LAYER_COUNT = 24
KV_HEADS = 8
HEAD_DIM = 128
PREFILL_LENGTH = 313


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture reusable per-layer references for a range of "
            "single-token autoregressive decode steps."
        )
    )
    parser.add_argument("--start-step", type=int, default=4)
    parser.add_argument("--end-step", type=int, default=10)
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dtype_tag(value: torch.Tensor) -> str:
    return {
        torch.float32: "f32",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.int64: "i64",
        torch.int32: "i32",
        torch.bool: "bool",
    }.get(value.dtype, str(value.dtype).replace("torch.", ""))


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().cpu().contiguous()
    numeric = detached.float()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "minimum": float(numeric.min().item()),
        "maximum": float(numeric.max().item()),
        "mean": float(numeric.mean().item()),
    }


def save_tensor(
    step: int,
    layer_index: int,
    name: str,
    value: torch.Tensor,
) -> dict[str, Any]:
    detached = value.detach().cpu().contiguous()
    decode_name = (
        f"decoder_layer{layer_index}_decode"
        if step == 1
        else f"decoder_layer{layer_index}_decode_step{step}"
    )
    npy_dir = REFERENCE_ROOT / decode_name
    raw_dir = ARTIFACTS_DIR / decode_name / "reference"
    npy_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    array = detached.numpy()
    npy_path = npy_dir / f"{name}.npy"
    raw_path = raw_dir / f"{name}_{dtype_tag(detached)}.bin"
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


def save_tail(
    step: int,
    hidden: torch.Tensor,
    norm_output: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    output_dir = ARTIFACTS_DIR / f"decode_tail_step{step}/reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "final_norm_input_f32.bin": hidden,
        "final_norm_output_f32.bin": norm_output,
        "decode_logits_f32.bin": logits,
    }
    for filename, value in tensors.items():
        value.detach().cpu().contiguous().numpy().astype(
            np.float32, copy=False
        ).tofile(output_dir / filename)


def main() -> None:
    capture_start = time.perf_counter()
    args = parse_args()
    reference_document = json.loads(
        GENERATED_TOKENS_PATH.read_text(encoding="utf-8")
    )
    expected_tokens = [
        int(token) for token in (
            reference_document["generated_token_ids"]
            if isinstance(reference_document, dict)
            else reference_document
        )
    ]
    if not 1 <= args.start_step <= args.end_step < len(expected_tokens):
        raise ValueError(
            "Step range must select decode outputs after the prefill token; "
            f"got {args.start_step}..{args.end_step} for "
            f"{len(expected_tokens)} generated tokens"
        )

    logging.disable_progress_bar()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    base = load_module(BASE_CAPTURE_SCRIPT, "autoregressive_capture_base")
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

    active_captures: dict[str, torch.Tensor] | None = None
    handles: list[Any] = []

    def make_pre_hook(layer_index: int) -> Callable[..., None]:
        prefix = f"layer{layer_index}"

        def hook(
            module: torch.nn.Module,
            hook_args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            if active_captures is None:
                return
            arguments = base.bind_forward_arguments(module, hook_args, kwargs)
            for name in ("hidden_states", "attention_mask", "position_embeddings"):
                if name not in arguments:
                    raise RuntimeError(f"Layer {layer_index} missing {name}")
                base.clone_nested_tensors(
                    f"{prefix}_{name}", arguments[name], active_captures
                )

        return hook

    def make_post_hook(layer_index: int) -> Callable[..., None]:
        prefix = f"layer{layer_index}"

        def hook(
            module: torch.nn.Module,
            hook_args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, hook_args, kwargs
            if active_captures is not None:
                base.clone_nested_tensors(
                    f"{prefix}_output", output, active_captures
                )

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

    report_steps: list[dict[str, Any]] = []
    full_input_ids = model_inputs["input_ids"]
    full_attention_mask = model_inputs["attention_mask"]
    full_mm_types = model_inputs["mm_token_type_ids"]

    try:
        with torch.inference_mode():
            output = model(
                **model_inputs,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        previous_token = output.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        if int(previous_token.item()) != expected_tokens[0]:
            raise RuntimeError("Prefill token does not match the golden run")

        for step in range(1, args.end_step + 1):
            full_input_ids = torch.cat((full_input_ids, previous_token), dim=1)
            full_attention_mask = torch.cat(
                (full_attention_mask, torch.ones_like(previous_token)), dim=1
            )
            full_mm_types = torch.cat(
                (
                    full_mm_types,
                    full_mm_types.new_zeros((full_mm_types.shape[0], 1)),
                ),
                dim=1,
            )

            with torch.inference_mode():
                prepared = model.prepare_inputs_for_generation(
                    input_ids=full_input_ids,
                    next_sequence_length=1,
                    past_key_values=output.past_key_values,
                    attention_mask=full_attention_mask,
                    mm_token_type_ids=full_mm_types,
                    pixel_values=model_inputs["pixel_values"],
                    image_grid_thw=model_inputs["image_grid_thw"],
                    use_cache=True,
                )

            expected_past_length = PREFILL_LENGTH + step - 1
            expected_present_length = PREFILL_LENGTH + step
            if prepared["attention_mask"].shape[-1] != expected_present_length:
                raise RuntimeError(f"Step {step} attention-mask length mismatch")
            if int(prepared["input_ids"].item()) != expected_tokens[step - 1]:
                raise RuntimeError(f"Step {step} input token mismatch")

            past_legacy = contract.cache_to_legacy(output.past_key_values)
            captures: dict[str, torch.Tensor] = {}
            active_captures = captures if step >= args.start_step else None
            step_start = time.perf_counter()
            step_arguments = dict(prepared)
            step_arguments.update(
                {"use_cache": True, "return_dict": True, "logits_to_keep": 1}
            )
            with torch.inference_mode():
                next_output = model(**step_arguments)
            decode_seconds = time.perf_counter() - step_start
            active_captures = None

            next_token = next_output.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            if int(next_token.item()) != expected_tokens[step]:
                raise RuntimeError(
                    f"Step {step} output token {int(next_token.item())} != "
                    f"{expected_tokens[step]}"
                )

            if step >= args.start_step:
                present_legacy = contract.cache_to_legacy(
                    next_output.past_key_values
                )
                expected_hidden = model.get_input_embeddings()(
                    previous_token
                ).detach().cpu().contiguous()
                actual_hidden = captures[
                    "layer0_hidden_states"
                ].detach().cpu().contiguous()
                embedding_error = float(
                    (actual_hidden - expected_hidden).abs().max().item()
                )
                if embedding_error != 0.0:
                    raise RuntimeError(f"Step {step} embedding handoff changed")

                maximum_boundary_error = 0.0
                for layer_index in range(LAYER_COUNT - 1):
                    boundary_error = float(
                        (
                            captures[f"layer{layer_index}_output"]
                            - captures[f"layer{layer_index + 1}_hidden_states"]
                        )
                        .abs()
                        .max()
                        .item()
                    )
                    maximum_boundary_error = max(
                        maximum_boundary_error, boundary_error
                    )
                if maximum_boundary_error != 0.0:
                    raise RuntimeError(f"Step {step} hidden boundary changed")

                layer_reports: list[dict[str, Any]] = []
                maximum_key_prefix_error = 0.0
                maximum_value_prefix_error = 0.0
                for layer_index in range(LAYER_COUNT):
                    prefix = f"layer{layer_index}"
                    past_key, past_value = (
                        value.detach().cpu().contiguous()
                        for value in past_legacy[layer_index]
                    )
                    present_key, present_value = (
                        value.detach().cpu().contiguous()
                        for value in present_legacy[layer_index]
                    )
                    if tuple(past_key.shape) != (
                        1,
                        KV_HEADS,
                        expected_past_length,
                        HEAD_DIM,
                    ):
                        raise RuntimeError(f"Step {step} layer {layer_index} past shape")
                    if tuple(present_key.shape) != (
                        1,
                        KV_HEADS,
                        expected_present_length,
                        HEAD_DIM,
                    ):
                        raise RuntimeError(
                            f"Step {step} layer {layer_index} present shape"
                        )
                    key_prefix_error = float(
                        (
                            present_key[:, :, :expected_past_length, :] - past_key
                        )
                        .abs()
                        .max()
                        .item()
                    )
                    value_prefix_error = float(
                        (
                            present_value[:, :, :expected_past_length, :]
                            - past_value
                        )
                        .abs()
                        .max()
                        .item()
                    )
                    if key_prefix_error != 0.0 or value_prefix_error != 0.0:
                        raise RuntimeError(
                            f"Step {step} layer {layer_index} KV prefix changed"
                        )
                    maximum_key_prefix_error = max(
                        maximum_key_prefix_error, key_prefix_error
                    )
                    maximum_value_prefix_error = max(
                        maximum_value_prefix_error, value_prefix_error
                    )

                    tensors = {
                        f"{prefix}_hidden_states": captures[
                            f"{prefix}_hidden_states"
                        ],
                        f"{prefix}_attention_mask": captures[
                            f"{prefix}_attention_mask"
                        ],
                        f"{prefix}_position_embeddings_0": captures[
                            f"{prefix}_position_embeddings_0"
                        ],
                        f"{prefix}_position_embeddings_1": captures[
                            f"{prefix}_position_embeddings_1"
                        ],
                        f"{prefix}_output": captures[f"{prefix}_output"],
                        "past_key": past_key,
                        "past_value": past_value,
                        "present_key": present_key,
                        "present_value": present_value,
                        "prepared_input_ids": prepared["input_ids"],
                        "prepared_attention_mask": prepared["attention_mask"],
                        "prepared_mm_token_type_ids": prepared[
                            "mm_token_type_ids"
                        ],
                        f"step{step}_input_token_id": previous_token,
                        "next_token_id": next_token,
                        "decode_logits": next_output.logits,
                    }
                    tensor_reports = {
                        name: save_tensor(step, layer_index, name, value)
                        for name, value in sorted(tensors.items())
                    }
                    layer_reports.append(
                        {
                            "layer_index": layer_index,
                            "key_prefix_max_abs_error": key_prefix_error,
                            "value_prefix_max_abs_error": value_prefix_error,
                            "tensors": tensor_reports,
                        }
                    )

                final_hidden = captures["layer23_output"]
                with torch.inference_mode():
                    norm_output = model.model.language_model.norm(final_hidden)
                    tail_logits = model.lm_head(norm_output)
                logits_error = float(
                    (tail_logits - next_output.logits).abs().max().item()
                )
                if logits_error > 1.0e-5:
                    raise RuntimeError(f"Step {step} tail logits changed")
                save_tail(step, final_hidden, norm_output, tail_logits)

                report_steps.append(
                    {
                        "step": step,
                        "input_token": int(previous_token.item()),
                        "output_token": int(next_token.item()),
                        "past_length": expected_past_length,
                        "present_length": expected_present_length,
                        "decode_seconds": decode_seconds,
                        "embedding_hidden_max_abs_error": embedding_error,
                        "maximum_hidden_boundary_error": maximum_boundary_error,
                        "maximum_key_prefix_error": maximum_key_prefix_error,
                        "maximum_value_prefix_error": maximum_value_prefix_error,
                        "tail_logits_max_abs_error": logits_error,
                        "layers": layer_reports,
                    }
                )
                print(
                    f"Step {step}: token {int(previous_token.item())} -> "
                    f"{int(next_token.item())}, KV {expected_past_length} -> "
                    f"{expected_present_length}, {decode_seconds:.3f}s"
                )

            output = next_output
            previous_token = next_token
    finally:
        active_captures = None
        for handle in handles:
            handle.remove()

    report_path = (
        DOCS_DIR
        / f"decoder_24_layer_decode_steps{args.start_step}_{args.end_step}_reference.json"
    )
    report = {
        "model_revision": (
            PROJECT_DIR / "configs/model_revision.txt"
        ).read_text(encoding="utf-8").strip(),
        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": model_load_seconds,
        "total_capture_seconds": time.perf_counter() - capture_start,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "expected_tokens": expected_tokens,
        "eos_token": expected_tokens[-1],
        "steps": report_steps,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    print("Captured autoregressive references successfully.")


if __name__ == "__main__":
    main()
