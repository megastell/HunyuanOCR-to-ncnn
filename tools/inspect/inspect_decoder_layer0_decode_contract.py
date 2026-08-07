from __future__ import annotations

import argparse
import inspect
import json
import os
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
    Path(os.environ.get(
        "HUNYUANOCR_MODEL_DIR",
        str(Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"),
    ))
)

REFERENCE_DIR = (
    Path(os.environ.get(
        "HUNYUANOCR_REFERENCE_DIR",
        str(PROJECT_DIR / "reference/smoke_en_cpu_fp32"),
    ))
)
DOCS_DIR = Path(os.environ.get("HUNYUANOCR_DOCS_DIR", str(PROJECT_DIR / "docs")))

REPORT_PATH = (
    DOCS_DIR / "decoder_layer0_decode_contract_probe.json"
)

TORCH_THREADS = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the single-token decode contract "
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


def load_tensor(name: str) -> torch.Tensor:
    path = REFERENCE_DIR / f"{name}.npy"

    if not path.is_file():
        raise FileNotFoundError(path)

    return torch.from_numpy(
        np.load(path)
    )


def tensor_info(
    value: torch.Tensor,
) -> dict[str, Any]:
    value_float = value.detach().float()

    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
        "minimum": float(
            value_float.min().item()
        ),
        "maximum": float(
            value_float.max().item()
        ),
        "mean": float(
            value_float.mean().item()
        ),
    }


def cache_to_legacy(
    cache: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if cache is None:
        raise RuntimeError("past_key_values为空。")

    if hasattr(cache, "to_legacy_cache"):
        legacy = cache.to_legacy_cache()

        return tuple(
            (key, value)
            for key, value in legacy
        )

    if (
        hasattr(cache, "key_cache")
        and hasattr(cache, "value_cache")
    ):
        return tuple(
            (key, value)
            for key, value in zip(
                cache.key_cache,
                cache.value_cache,
            )
        )

    if hasattr(cache, "layers"):
        result: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []

        for index, layer in enumerate(
            cache.layers
        ):
            key = None
            value = None

            for attribute in (
                "keys",
                "key",
                "key_cache",
            ):
                if hasattr(layer, attribute):
                    key = getattr(layer, attribute)
                    break

            for attribute in (
                "values",
                "value",
                "value_cache",
            ):
                if hasattr(layer, attribute):
                    value = getattr(layer, attribute)
                    break

            if not isinstance(key, torch.Tensor):
                raise RuntimeError(
                    f"第{index}层无法获取key cache。"
                )

            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"第{index}层无法获取value cache。"
                )

            result.append((key, value))

        return tuple(result)

    try:
        return tuple(
            (key, value)
            for key, value in cache
        )
    except Exception as error:
        raise RuntimeError(
            "无法将cache转换为逐层K/V结构。"
        ) from error


def describe_mapping(
    mapping: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    for key, value in mapping.items():
        if isinstance(value, torch.Tensor):
            report[key] = tensor_info(value)

        elif value is None:
            report[key] = None

        else:
            report[key] = {
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

    global REPORT_PATH

    REPORT_PATH = DOCS_DIR / f"decoder_layer{layer_index}_decode_contract_probe.json"

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    model_inputs = {
        name: load_tensor(name)
        for name in (
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        )
    }

    print("===== Model methods =====")

    print(
        "forward:",
        inspect.signature(
            HunYuanVLForConditionalGeneration.forward
        ),
    )

    print(
        "prepare_inputs_for_generation:",
        inspect.signature(
            HunYuanVLForConditionalGeneration
            .prepare_inputs_for_generation
        ),
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

    print()
    print("===== Prefill with cache =====")

    prefill_start = time.perf_counter()

    with torch.inference_mode():
        prefill_output = model(
            **model_inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

    prefill_seconds = (
        time.perf_counter() - prefill_start
    )

    prefill_cache = (
        prefill_output.past_key_values
    )

    prefill_legacy = cache_to_legacy(
        prefill_cache
    )

    if len(prefill_legacy) != 24:
        raise RuntimeError(
            "Decoder层数错误："
            f"实际={len(prefill_legacy)}，预期=24"
        )

    if layer_index >= len(prefill_legacy):
        raise ValueError(
            f"layer_index={layer_index}越界，"
            f"模型缓存共有{len(prefill_legacy)}层。"
        )

    prefill_key0 = (
        prefill_legacy[layer_index][0]
        .detach()
        .clone()
    )

    prefill_value0 = (
        prefill_legacy[layer_index][1]
        .detach()
        .clone()
    )

    next_token_id = (
        prefill_output.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    print(
        "cache type:",
        type(prefill_cache).__name__,
    )

    print(
        "cache layers:",
        len(prefill_legacy),
    )

    print(
        f"layer{layer_index} key:",
        tuple(prefill_key0.shape),
    )

    print(
        f"layer{layer_index} value:",
        tuple(prefill_value0.shape),
    )

    print(
        "next token id:",
        next_token_id.tolist(),
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

    # 新生成的纯文本token使用mm_token_type_id=0。
    # prepare_inputs_for_generation会保留完整历史类型序列。
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

    prepare_arguments = {
        "input_ids": full_input_ids,

        # 只将最后一个新token送入模型。
        # 不提供该参数时，完整的314个token都会被视为新输入。
        "next_sequence_length": 1,

        "past_key_values": prefill_cache,
        "attention_mask":
            extended_attention_mask,
        "mm_token_type_ids":
            extended_mm_token_type_ids,
        "pixel_values":
            model_inputs["pixel_values"],
        "image_grid_thw":
            model_inputs["image_grid_thw"],
        "use_cache": True,
    }

    print()
    print("===== Prepare decode inputs =====")

    with torch.inference_mode():
        prepared_inputs = (
            model.prepare_inputs_for_generation(
                **prepare_arguments
            )
        )

    print("prepared keys:")

    for key, value in prepared_inputs.items():
        if isinstance(value, torch.Tensor):
            print(
                f"  {key:28s}",
                tuple(value.shape),
                value.dtype,
            )
        else:
            print(
                f"  {key:28s}",
                type(value).__name__,
            )

    print()
    print("===== Single-token decode =====")

    decode_start = time.perf_counter()

    decode_arguments = dict(prepared_inputs)
    decode_arguments["use_cache"] = True
    decode_arguments["return_dict"] = True
    decode_arguments["logits_to_keep"] = 1

    with torch.inference_mode():
        decode_output = model(
            **decode_arguments
        )

    decode_seconds = (
        time.perf_counter() - decode_start
    )

    decode_cache = (
        decode_output.past_key_values
    )

    decode_legacy = cache_to_legacy(
        decode_cache
    )

    if layer_index >= len(decode_legacy):
        raise ValueError(
            f"layer_index={layer_index}越界，"
            f"Decode缓存共有{len(decode_legacy)}层。"
        )

    decode_key0 = (
        decode_legacy[layer_index][0]
        .detach()
        .clone()
    )

    decode_value0 = (
        decode_legacy[layer_index][1]
        .detach()
        .clone()
    )

    print(
        "decode logits:",
        tuple(decode_output.logits.shape),
    )

    print(
        f"present layer{layer_index} key:",
        tuple(decode_key0.shape),
    )

    print(
        f"present layer{layer_index} value:",
        tuple(decode_value0.shape),
    )

    if tuple(prefill_key0.shape) != (
        1,
        8,
        313,
        128,
    ):
        raise RuntimeError(
            "Prefill key形状不符合预期："
            f"{tuple(prefill_key0.shape)}"
        )

    if tuple(prefill_value0.shape) != (
        1,
        8,
        313,
        128,
    ):
        raise RuntimeError(
            "Prefill value形状不符合预期："
            f"{tuple(prefill_value0.shape)}"
        )

    if tuple(decode_key0.shape) != (
        1,
        8,
        314,
        128,
    ):
        raise RuntimeError(
            "Present key形状不符合预期："
            f"{tuple(decode_key0.shape)}"
        )

    if tuple(decode_value0.shape) != (
        1,
        8,
        314,
        128,
    ):
        raise RuntimeError(
            "Present value形状不符合预期："
            f"{tuple(decode_value0.shape)}"
        )

    key_prefix_error = (
        decode_key0[:, :, :313, :]
        - prefill_key0
    ).abs()

    value_prefix_error = (
        decode_value0[:, :, :313, :]
        - prefill_value0
    ).abs()

    print()
    print("===== Cache append checks =====")

    print(
        "key prefix max error:",
        float(key_prefix_error.max().item()),
    )

    print(
        "value prefix max error:",
        float(
            value_prefix_error.max().item()
        ),
    )

    report = {
        "model_revision": (
            PROJECT_DIR
            / "configs/model_revision.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),
        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,
        "layer_index": layer_index,
        "model_load_seconds": load_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "cache_type": type(
            prefill_cache
        ).__name__,
        "cache_layer_count": len(
            prefill_legacy
        ),
        "next_token_id": int(
            next_token_id.item()
        ),
        "prefill_logits":
            tensor_info(
                prefill_output.logits
            ),
        f"prefill_layer{layer_index}_key":
            tensor_info(prefill_key0),
        f"prefill_layer{layer_index}_value":
            tensor_info(prefill_value0),
        "full_input_ids":
            tensor_info(full_input_ids),
        "extended_attention_mask":
            tensor_info(
                extended_attention_mask
            ),
        "extended_mm_token_type_ids":
            tensor_info(
                extended_mm_token_type_ids
            ),
        "prepared_inputs":
            describe_mapping(prepared_inputs),
        "decode_logits":
            tensor_info(
                decode_output.logits
            ),
        f"present_layer{layer_index}_key":
            tensor_info(decode_key0),
        f"present_layer{layer_index}_value":
            tensor_info(decode_value0),
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
        (
            f"✅ Decoder Layer {layer_index} "
            "单Token与KV Cache契约确认成功。"
        )
    )


if __name__ == "__main__":
    main()
