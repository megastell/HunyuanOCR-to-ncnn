from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import HunYuanVLForConditionalGeneration


PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

MODEL_DIR = (
    Path.home()
    / "work/hunyuanocr/models/HunyuanOCR-1.5"
)

CONTRACT_SCRIPT = (
    PROJECT_DIR
    / "tools/inspect"
    / "inspect_decoder_layer0_decode_contract.py"
)

REFERENCE_DIR = (
    PROJECT_DIR
    / "reference/smoke_en_cpu_fp32"
    / "decoder_layer0_decode"
)

RAW_DIR = (
    PROJECT_DIR
    / "artifacts/decoder_layer0_decode"
    / "reference"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/decoder_layer0_decode_reference.json"
)

TORCH_THREADS = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the single-token decode reference "
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


def load_contract_module():
    specification = (
        importlib.util.spec_from_file_location(
            "decoder_decode_contract",
            CONTRACT_SCRIPT,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "无法加载Decode契约探针模块。"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


def dtype_tag(value: torch.Tensor) -> str:
    mapping = {
        torch.float32: "f32",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.int64: "i64",
        torch.int32: "i32",
        torch.bool: "bool",
    }

    return mapping.get(
        value.dtype,
        str(value.dtype).replace("torch.", ""),
    )


def tensor_summary(
    value: torch.Tensor,
) -> dict[str, Any]:
    detached = value.detach().cpu()

    numeric = detached.float()
    finite_mask = torch.isfinite(numeric)
    finite_values = numeric[finite_mask]

    if finite_values.numel() > 0:
        minimum = float(
            finite_values.min().item()
        )

        maximum = float(
            finite_values.max().item()
        )

        mean = float(
            finite_values.mean().item()
        )
    else:
        minimum = None
        maximum = None
        mean = None

    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "minimum_finite": minimum,
        "maximum_finite": maximum,
        "mean_finite": mean,
        "nan_count": int(
            torch.isnan(numeric).sum().item()
        ),
        "positive_inf_count": int(
            torch.isposinf(numeric).sum().item()
        ),
        "negative_inf_count": int(
            torch.isneginf(numeric).sum().item()
        ),
    }


def save_tensor(
    name: str,
    value: torch.Tensor,
) -> dict[str, Any]:
    detached = (
        value.detach()
        .cpu()
        .contiguous()
    )

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    npy_path = REFERENCE_DIR / f"{name}.npy"

    array = detached.numpy()
    np.save(npy_path, array)

    tag = dtype_tag(detached)
    raw_path = RAW_DIR / f"{name}_{tag}.bin"

    array.tofile(raw_path)

    report = tensor_summary(detached)

    report.update(
        {
            "npy_path":
                npy_path.relative_to(
                    PROJECT_DIR
                ).as_posix(),

            "raw_path":
                raw_path.relative_to(
                    PROJECT_DIR
                ).as_posix(),

            "raw_bytes":
                raw_path.stat().st_size,
        }
    )

    return report


def clone_nested_tensors(
    prefix: str,
    value: Any,
    destination: dict[str, torch.Tensor],
) -> None:
    if isinstance(value, torch.Tensor):
        destination[prefix] = (
            value.detach()
            .cpu()
            .clone()
        )

        return

    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            clone_nested_tensors(
                f"{prefix}_{index}",
                item,
                destination,
            )

        return

    if isinstance(value, dict):
        for key, item in value.items():
            clone_nested_tensors(
                f"{prefix}_{key}",
                item,
                destination,
            )


def bind_forward_arguments(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(
        module.forward
    )

    bound = signature.bind_partial(
        *args,
        **kwargs,
    )

    return dict(bound.arguments)


def describe_non_tensor_arguments(
    arguments: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    for name, value in arguments.items():
        if isinstance(value, torch.Tensor):
            continue

        if isinstance(value, (tuple, list)):
            if any(
                isinstance(item, torch.Tensor)
                for item in value
            ):
                continue

        if isinstance(value, dict):
            if any(
                isinstance(item, torch.Tensor)
                for item in value.values()
            ):
                continue

        report[name] = {
            "type": type(value).__name__,
            "repr": repr(value)[:500],
        }

    return report


def main() -> None:
    args = parse_args()
    layer_index = args.layer_index

    if layer_index < 0:
        raise ValueError(
            f"layer_index不能为负数：{layer_index}"
        )

    layer_prefix = f"layer{layer_index}"
    decode_name = f"decoder_layer{layer_index}_decode"

    global REFERENCE_DIR
    global RAW_DIR
    global REPORT_PATH

    REFERENCE_DIR = (
        PROJECT_DIR
        / "reference/smoke_en_cpu_fp32"
        / decode_name
    )

    RAW_DIR = (
        PROJECT_DIR
        / "artifacts"
        / decode_name
        / "reference"
    )

    REPORT_PATH = (
        PROJECT_DIR
        / "docs"
        / f"{decode_name}_reference.json"
    )

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    contract = load_contract_module()

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
        time.perf_counter() - load_start
    )

    print(
        f"Model loaded in {load_seconds:.3f}s"
    )

    print()
    print("===== Prefill =====")

    with torch.inference_mode():
        prefill_output = model(
            **model_inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

    prefill_cache = (
        prefill_output.past_key_values
    )

    prefill_legacy = contract.cache_to_legacy(
        prefill_cache
    )

    prefill_key0 = (
        prefill_legacy[layer_index][0]
        .detach()
        .cpu()
        .clone()
    )

    prefill_value0 = (
        prefill_legacy[layer_index][1]
        .detach()
        .cpu()
        .clone()
    )

    next_token_id = (
        prefill_output.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    full_input_ids = torch.cat(
        (
            model_inputs["input_ids"],
            next_token_id,
        ),
        dim=1,
    )

    extended_attention_mask = torch.cat(
        (
            model_inputs["attention_mask"],
            torch.ones_like(next_token_id),
        ),
        dim=1,
    )

    extended_mm_token_type_ids = torch.cat(
        (
            model_inputs["mm_token_type_ids"],

            model_inputs[
                "mm_token_type_ids"
            ].new_zeros(
                (
                    model_inputs[
                        "mm_token_type_ids"
                    ].shape[0],
                    1,
                )
            ),
        ),
        dim=1,
    )

    with torch.inference_mode():
        prepared_inputs = (
            model.prepare_inputs_for_generation(
                input_ids=full_input_ids,
                next_sequence_length=1,
                past_key_values=prefill_cache,
                attention_mask=
                    extended_attention_mask,
                mm_token_type_ids=
                    extended_mm_token_type_ids,
                pixel_values=
                    model_inputs["pixel_values"],
                image_grid_thw=
                    model_inputs["image_grid_thw"],
                use_cache=True,
            )
        )

    if tuple(
        prepared_inputs["input_ids"].shape
    ) != (1, 1):
        raise RuntimeError(
            "Decode input_ids形状错误："
            f"{tuple(prepared_inputs['input_ids'].shape)}"
        )

    decoder_layer = (
        model.model
        .language_model
        .layers[layer_index]
    )

    self_attention = decoder_layer.self_attn

    captures: dict[str, torch.Tensor] = {}

    argument_structure: dict[str, Any] = {}

    def layer_pre_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        arguments = bind_forward_arguments(
            module,
            args,
            kwargs,
        )

        argument_structure[
            f"{layer_prefix}_forward_non_tensor"
        ] = describe_non_tensor_arguments(
            arguments
        )

        for name, value in arguments.items():
            clone_nested_tensors(
                f"{layer_prefix}_{name}",
                value,
                captures,
            )

    def attention_pre_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        arguments = bind_forward_arguments(
            module,
            args,
            kwargs,
        )

        argument_structure[
            "self_attention_forward_non_tensor"
        ] = describe_non_tensor_arguments(
            arguments
        )

        for name, value in arguments.items():
            clone_nested_tensors(
                f"self_attention_{name}",
                value,
                captures,
            )

    def layer_post_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        del module
        del args
        del kwargs

        clone_nested_tensors(
            f"{layer_prefix}_output",
            output,
            captures,
        )

    handles = [
        decoder_layer.register_forward_pre_hook(
            layer_pre_hook,
            with_kwargs=True,
        ),

        self_attention.register_forward_pre_hook(
            attention_pre_hook,
            with_kwargs=True,
        ),

        decoder_layer.register_forward_hook(
            layer_post_hook,
            with_kwargs=True,
        ),
    ]

    print()
    print("===== Single-token decode capture =====")

    decode_start = time.perf_counter()

    decode_arguments = dict(prepared_inputs)

    decode_arguments.update(
        {
            "use_cache": True,
            "return_dict": True,
            "logits_to_keep": 1,
        }
    )

    with torch.inference_mode():
        decode_output = model(
            **decode_arguments
        )

    decode_seconds = (
        time.perf_counter()
        - decode_start
    )

    for handle in handles:
        handle.remove()

    decode_legacy = contract.cache_to_legacy(
        decode_output.past_key_values
    )

    present_key0 = (
        decode_legacy[layer_index][0]
        .detach()
        .cpu()
        .clone()
    )

    present_value0 = (
        decode_legacy[layer_index][1]
        .detach()
        .cpu()
        .clone()
    )

    if tuple(prefill_key0.shape) != (
        1,
        8,
        313,
        128,
    ):
        raise RuntimeError(
            f"past key形状错误："
            f"{tuple(prefill_key0.shape)}"
        )

    if tuple(present_key0.shape) != (
        1,
        8,
        314,
        128,
    ):
        raise RuntimeError(
            f"present key形状错误："
            f"{tuple(present_key0.shape)}"
        )

    key_prefix_error = (
        present_key0[:, :, :313, :]
        - prefill_key0
    ).abs()

    value_prefix_error = (
        present_value0[:, :, :313, :]
        - prefill_value0
    ).abs()

    if float(key_prefix_error.max()) != 0.0:
        raise RuntimeError(
            "Key cache历史前缀发生变化。"
        )

    if float(value_prefix_error.max()) != 0.0:
        raise RuntimeError(
            "Value cache历史前缀发生变化。"
        )

    captures.update(
        {
            "prepared_input_ids":
                prepared_inputs[
                    "input_ids"
                ].detach().cpu().clone(),

            "prepared_attention_mask":
                prepared_inputs[
                    "attention_mask"
                ].detach().cpu().clone(),

            "prepared_mm_token_type_ids":
                prepared_inputs[
                    "mm_token_type_ids"
                ].detach().cpu().clone(),

            "next_token_id":
                next_token_id.detach()
                .cpu().clone(),

            "past_key":
                prefill_key0,

            "past_value":
                prefill_value0,

            "present_key":
                present_key0,

            "present_value":
                present_value0,

            "decode_logits":
                decode_output.logits
                .detach().cpu().clone(),
        }
    )

    print("Captured tensors:")

    for name in sorted(captures):
        value = captures[name]

        print(
            f"  {name:42s} "
            f"{tuple(value.shape)!s:24s} "
            f"{value.dtype}"
        )

    reports: dict[str, Any] = {}

    for name, value in sorted(
        captures.items()
    ):
        reports[name] = save_tensor(
            name,
            value,
        )

    report = {
        "layer_index": layer_index,

        "model_revision": (
            PROJECT_DIR
            / "configs/model_revision.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),

        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": load_seconds,
        "decode_seconds": decode_seconds,

        "contract": {
            "past_key_shape":
                list(prefill_key0.shape),

            "past_value_shape":
                list(prefill_value0.shape),

            "prepared_input_ids_shape":
                list(
                    prepared_inputs[
                        "input_ids"
                    ].shape
                ),

            "prepared_attention_mask_shape":
                list(
                    prepared_inputs[
                        "attention_mask"
                    ].shape
                ),

            "prepared_mm_token_type_ids_shape":
                list(
                    prepared_inputs[
                        "mm_token_type_ids"
                    ].shape
                ),

            "present_key_shape":
                list(present_key0.shape),

            "present_value_shape":
                list(present_value0.shape),
        },

        "cache_checks": {
            "key_prefix_max_abs_error":
                float(
                    key_prefix_error.max().item()
                ),

            "value_prefix_max_abs_error":
                float(
                    value_prefix_error.max().item()
                ),
        },

        "module_argument_structure":
            argument_structure,

        "tensors": reports,
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
    print(
        "Key prefix max error:",
        float(key_prefix_error.max()),
    )

    print(
        "Value prefix max error:",
        float(value_prefix_error.max()),
    )

    print("Report:", REPORT_PATH)

    print(
        f"✅ Decoder Layer {layer_index} Decode参考捕获成功。"
    )


if __name__ == "__main__":
    main()
