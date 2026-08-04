from __future__ import annotations

import argparse
import importlib.util
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

BASE_CAPTURE_SCRIPT = (
    PROJECT_DIR
    / "tools/reference"
    / "capture_decoder_layer0_decode_reference.py"
)

GENERATED_TOKENS_PATH = (
    PROJECT_DIR
    / "reference/smoke_en_cpu_fp32"
    / "generated_token_ids.json"
)

TORCH_THREADS = 9
HIDDEN_SIZE = 1024
KV_HEADS = 8
HEAD_DIM = 128
PREFILL_LENGTH = 313
STEP1_PRESENT_LENGTH = 314
STEP2_PRESENT_LENGTH = 315


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the second single-token decode "
            "reference for a selected HunyuanOCR layer."
        )
    )

    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
    )

    return parser.parse_args()


def load_base_module():
    specification = (
        importlib.util.spec_from_file_location(
            "decoder_decode_reference_base",
            BASE_CAPTURE_SCRIPT,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "无法加载第一步Decode参考捕获模块。"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


def tensor_summary(
    value: torch.Tensor,
) -> dict[str, Any]:
    detached = (
        value.detach()
        .cpu()
        .contiguous()
    )

    numeric = detached.float()
    finite = numeric[torch.isfinite(numeric)]

    if finite.numel() > 0:
        minimum = float(finite.min().item())
        maximum = float(finite.max().item())
        mean = float(finite.mean().item())
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
    }


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


def main() -> None:
    arguments = parse_args()
    layer_index = arguments.layer_index

    if layer_index < 0 or layer_index > 23:
        raise ValueError(
            f"layer-index必须在0到23之间：{layer_index}"
        )

    layer_prefix = f"layer{layer_index}"

    reference_dir = (
        PROJECT_DIR
        / "reference/smoke_en_cpu_fp32"
        / f"decoder_layer{layer_index}_decode_step2"
    )

    raw_dir = (
        PROJECT_DIR
        / f"artifacts/decoder_layer{layer_index}_decode_step2"
        / "reference"
    )

    report_path = (
        PROJECT_DIR
        / "docs"
        / f"decoder_layer{layer_index}_decode_step2_reference.json"
    )

    step1_reference_dir = (
        PROJECT_DIR
        / "reference/smoke_en_cpu_fp32"
        / f"decoder_layer{layer_index}_decode"
    )

    def save_tensor(
        name: str,
        value: torch.Tensor,
    ) -> dict[str, Any]:
        detached = (
            value.detach()
            .cpu()
            .contiguous()
        )

        reference_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        npy_path = reference_dir / f"{name}.npy"

        array = detached.numpy()
        np.save(npy_path, array)

        tag = dtype_tag(detached)
        raw_path = raw_dir / f"{name}_{tag}.bin"
        array.tofile(raw_path)

        report = tensor_summary(detached)

        report.update(
            {
                "npy_path": (
                    npy_path.relative_to(
                        PROJECT_DIR
                    ).as_posix()
                ),
                "raw_path": (
                    raw_path.relative_to(
                        PROJECT_DIR
                    ).as_posix()
                ),
                "raw_bytes": raw_path.stat().st_size,
            }
        )

        return report

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    base = load_base_module()
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

    reference_tokens = json.loads(
        GENERATED_TOKENS_PATH.read_text(
            encoding="utf-8"
        )
    )

    if len(reference_tokens) < 3:
        raise RuntimeError(
            "参考生成序列不足3个token。"
        )

    expected_prefill_token = int(
        reference_tokens[0]
    )

    expected_step1_token = int(
        reference_tokens[1]
    )

    expected_step2_token = int(
        reference_tokens[2]
    )

    print("===== Token contract =====")
    print(
        "Prefill output token:",
        expected_prefill_token,
    )
    print(
        "Step 1 output token:",
        expected_step1_token,
    )
    print(
        "Step 2 output token:",
        expected_step2_token,
    )

    print()
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

    # --------------------------------------------------
    # Prefill: sequence length 313, predicts token 93892.
    # --------------------------------------------------

    print()
    print("===== Prefill =====")

    with torch.inference_mode():
        prefill_output = model(
            **model_inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

    prefill_token = (
        prefill_output.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    if int(prefill_token.item()) != (
        expected_prefill_token
    ):
        raise RuntimeError(
            "Prefill token与参考序列不一致。"
        )

    # --------------------------------------------------
    # First decode: input token 93892, predicts 5112.
    # --------------------------------------------------

    full_input_ids_step1 = torch.cat(
        (
            model_inputs["input_ids"],
            prefill_token,
        ),
        dim=1,
    )

    attention_mask_step1 = torch.cat(
        (
            model_inputs["attention_mask"],
            torch.ones_like(prefill_token),
        ),
        dim=1,
    )

    mm_types_step1 = torch.cat(
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
        prepared_step1 = (
            model.prepare_inputs_for_generation(
                input_ids=full_input_ids_step1,
                next_sequence_length=1,
                past_key_values=(
                    prefill_output.past_key_values
                ),
                attention_mask=attention_mask_step1,
                mm_token_type_ids=mm_types_step1,
                pixel_values=model_inputs[
                    "pixel_values"
                ],
                image_grid_thw=model_inputs[
                    "image_grid_thw"
                ],
                use_cache=True,
            )
        )

        step1_arguments = dict(
            prepared_step1
        )

        step1_arguments.update(
            {
                "use_cache": True,
                "return_dict": True,
                "logits_to_keep": 1,
            }
        )

        step1_output = model(
            **step1_arguments
        )

    step1_token = (
        step1_output.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    if int(step1_token.item()) != (
        expected_step1_token
    ):
        raise RuntimeError(
            "第一次Decode输出token与参考不一致。"
        )

    step1_legacy = contract.cache_to_legacy(
        step1_output.past_key_values
    )

    step2_past_key = (
        step1_legacy[layer_index][0]
        .detach()
        .cpu()
        .clone()
    )

    step2_past_value = (
        step1_legacy[layer_index][1]
        .detach()
        .cpu()
        .clone()
    )

    expected_step1_key_path = (
        step1_reference_dir
        / "present_key.npy"
    )

    expected_step1_value_path = (
        step1_reference_dir
        / "present_value.npy"
    )

    if (
        not expected_step1_key_path.is_file()
        or not expected_step1_value_path.is_file()
    ):
        raise FileNotFoundError(
            "找不到第一次Decode的present KV参考。"
        )

    expected_step1_key = torch.from_numpy(
        np.load(expected_step1_key_path)
    )

    expected_step1_value = torch.from_numpy(
        np.load(expected_step1_value_path)
    )

    step1_key_reference_error = float(
        (
            step2_past_key
            - expected_step1_key
        ).abs().max().item()
    )

    step1_value_reference_error = float(
        (
            step2_past_value
            - expected_step1_value
        ).abs().max().item()
    )

    if step1_key_reference_error != 0.0:
        raise RuntimeError(
            "Step2 past key与Step1 present key参考不一致。"
        )

    if step1_value_reference_error != 0.0:
        raise RuntimeError(
            "Step2 past value与Step1 present value参考不一致。"
        )

    # --------------------------------------------------
    # Second decode: input token 5112, predicts 206.
    # --------------------------------------------------

    full_input_ids_step2 = torch.cat(
        (
            full_input_ids_step1,
            step1_token,
        ),
        dim=1,
    )

    attention_mask_step2 = torch.cat(
        (
            attention_mask_step1,
            torch.ones_like(step1_token),
        ),
        dim=1,
    )

    mm_types_step2 = torch.cat(
        (
            mm_types_step1,
            mm_types_step1.new_zeros(
                (
                    mm_types_step1.shape[0],
                    1,
                )
            ),
        ),
        dim=1,
    )

    with torch.inference_mode():
        prepared_step2 = (
            model.prepare_inputs_for_generation(
                input_ids=full_input_ids_step2,
                next_sequence_length=1,
                past_key_values=(
                    step1_output.past_key_values
                ),
                attention_mask=attention_mask_step2,
                mm_token_type_ids=mm_types_step2,
                pixel_values=model_inputs[
                    "pixel_values"
                ],
                image_grid_thw=model_inputs[
                    "image_grid_thw"
                ],
                use_cache=True,
            )
        )

    if tuple(
        prepared_step2["input_ids"].shape
    ) != (1, 1):
        raise RuntimeError(
            "第二步prepared input_ids形状错误。"
        )

    if int(
        prepared_step2["input_ids"].item()
    ) != expected_step1_token:
        raise RuntimeError(
            "第二步输入token不是5112。"
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
        bound = base.bind_forward_arguments(
            module,
            args,
            kwargs,
        )

        argument_structure[
            f"{layer_prefix}_forward_non_tensor"
        ] = base.describe_non_tensor_arguments(
            bound
        )

        for name, value in bound.items():
            base.clone_nested_tensors(
                f"{layer_prefix}_{name}",
                value,
                captures,
            )

    def attention_pre_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        bound = base.bind_forward_arguments(
            module,
            args,
            kwargs,
        )

        argument_structure[
            "self_attention_forward_non_tensor"
        ] = base.describe_non_tensor_arguments(
            bound
        )

        for name, value in bound.items():
            base.clone_nested_tensors(
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
        del module, args, kwargs

        base.clone_nested_tensors(
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
    print("===== Second single-token decode capture =====")

    decode_start = time.perf_counter()

    with torch.inference_mode():
        step2_arguments = dict(
            prepared_step2
        )

        step2_arguments.update(
            {
                "use_cache": True,
                "return_dict": True,
                "logits_to_keep": 1,
            }
        )

        step2_output = model(
            **step2_arguments
        )

    decode_seconds = (
        time.perf_counter() - decode_start
    )

    for handle in handles:
        handle.remove()

    step2_token = (
        step2_output.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    if int(step2_token.item()) != (
        expected_step2_token
    ):
        raise RuntimeError(
            "第二次Decode输出token与参考不一致。"
        )

    step2_legacy = contract.cache_to_legacy(
        step2_output.past_key_values
    )

    present_key = (
        step2_legacy[layer_index][0]
        .detach()
        .cpu()
        .clone()
    )

    present_value = (
        step2_legacy[layer_index][1]
        .detach()
        .cpu()
        .clone()
    )

    expected_past_shape = (
        1,
        KV_HEADS,
        STEP1_PRESENT_LENGTH,
        HEAD_DIM,
    )

    expected_present_shape = (
        1,
        KV_HEADS,
        STEP2_PRESENT_LENGTH,
        HEAD_DIM,
    )

    if tuple(step2_past_key.shape) != (
        expected_past_shape
    ):
        raise RuntimeError(
            "第二步past key形状错误："
            f"{tuple(step2_past_key.shape)}"
        )

    if tuple(present_key.shape) != (
        expected_present_shape
    ):
        raise RuntimeError(
            "第二步present key形状错误："
            f"{tuple(present_key.shape)}"
        )

    key_prefix_error = float(
        (
            present_key[
                :,
                :,
                :STEP1_PRESENT_LENGTH,
                :,
            ]
            - step2_past_key
        ).abs().max().item()
    )

    value_prefix_error = float(
        (
            present_value[
                :,
                :,
                :STEP1_PRESENT_LENGTH,
                :,
            ]
            - step2_past_value
        ).abs().max().item()
    )

    if key_prefix_error != 0.0:
        raise RuntimeError(
            "第二步Key cache历史前缀发生变化。"
        )

    if value_prefix_error != 0.0:
        raise RuntimeError(
            "第二步Value cache历史前缀发生变化。"
        )

    captures.update(
        {
            "prepared_input_ids": (
                prepared_step2["input_ids"]
                .detach().cpu().clone()
            ),
            "prepared_attention_mask": (
                prepared_step2[
                    "attention_mask"
                ].detach().cpu().clone()
            ),
            "prepared_mm_token_type_ids": (
                prepared_step2[
                    "mm_token_type_ids"
                ].detach().cpu().clone()
            ),
            "step2_input_token_id": (
                step1_token.detach()
                .cpu().clone()
            ),
            "next_token_id": (
                step2_token.detach()
                .cpu().clone()
            ),
            "past_key": step2_past_key,
            "past_value": step2_past_value,
            "present_key": present_key,
            "present_value": present_value,
            "decode_logits": (
                step2_output.logits
                .detach().cpu().clone()
            ),
        }
    )

    print("Captured tensors:")

    for name in sorted(captures):
        value = captures[name]

        print(
            f"  {name:44s}"
            f"{tuple(value.shape)!s:25s}"
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
        "decode_step": 2,
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "token_contract": {
            "prefill_output_token": (
                expected_prefill_token
            ),
            "step1_output_token": (
                expected_step1_token
            ),
            "step2_output_token": (
                expected_step2_token
            ),
        },
        "contract": {
            "past_key_shape": list(
                step2_past_key.shape
            ),
            "past_value_shape": list(
                step2_past_value.shape
            ),
            "present_key_shape": list(
                present_key.shape
            ),
            "present_value_shape": list(
                present_value.shape
            ),
            "prepared_input_ids_shape": list(
                prepared_step2[
                    "input_ids"
                ].shape
            ),
            "prepared_attention_mask_shape": list(
                prepared_step2[
                    "attention_mask"
                ].shape
            ),
            "prepared_mm_token_type_ids_shape": list(
                prepared_step2[
                    "mm_token_type_ids"
                ].shape
            ),
        },
        "cache_checks": {
            "step1_present_vs_step2_past_key_max_error":
                step1_key_reference_error,
            "step1_present_vs_step2_past_value_max_error":
                step1_value_reference_error,
            "step2_key_prefix_max_error":
                key_prefix_error,
            "step2_value_prefix_max_error":
                value_prefix_error,
        },
        "module_argument_structure":
            argument_structure,
        "tensors": reports,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Step1 present -> Step2 past key error:",
        step1_key_reference_error,
    )
    print(
        "Step1 present -> Step2 past value error:",
        step1_value_reference_error,
    )
    print(
        "Step2 key history prefix error:",
        key_prefix_error,
    )
    print(
        "Step2 value history prefix error:",
        value_prefix_error,
    )
    print(
        "Step2 input token:",
        int(step1_token.item()),
    )
    print(
        "Step2 output token:",
        int(step2_token.item()),
    )
    print("Report:", report_path)

    print(
        f"✅ Decoder Layer {layer_index} "
        "第二次Decode参考捕获成功。"
    )


if __name__ == "__main__":
    main()
