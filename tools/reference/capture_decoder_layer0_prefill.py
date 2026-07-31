from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
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

REFERENCE_DIR = (
    PROJECT_DIR
    / "reference/smoke_en_cpu_fp32"
)

OUTPUT_DIR = (
    REFERENCE_DIR
    / "decoder_layer0_prefill"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/decoder_layer0_prefill_contract.json"
)

SEQUENCE_LENGTH = 313
HIDDEN_SIZE = 1024
TORCH_THREADS = 9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()

    if isinstance(value, tuple):
        return tuple(
            clone_value(item)
            for item in value
        )

    if isinstance(value, list):
        return [
            clone_value(item)
            for item in value
        ]

    return value


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


def save_tensor(
    name: str,
    tensor: torch.Tensor,
) -> dict[str, Any]:
    cpu_tensor = (
        tensor.detach()
        .cpu()
        .contiguous()
    )

    save_value = cpu_tensor

    if save_value.dtype == torch.bfloat16:
        save_value = save_value.float()

    path = OUTPUT_DIR / f"{name}.npy"

    np.save(
        path,
        save_value.numpy(),
    )

    numeric = cpu_tensor.float()
    finite_mask = torch.isfinite(numeric)

    report: dict[str, Any] = {
        "file": str(path),
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "numel": cpu_tensor.numel(),
        "sha256": sha256_file(path),
        "has_nan": bool(
            torch.isnan(numeric).any().item()
        ),
        "has_inf": bool(
            torch.isinf(numeric).any().item()
        ),
    }

    if finite_mask.any():
        finite = numeric[finite_mask]

        report["finite_min"] = float(
            finite.min().item()
        )

        report["finite_max"] = float(
            finite.max().item()
        )

        report["finite_mean"] = float(
            finite.mean().item()
        )

    print(
        f"{name:32s} "
        f"shape={tuple(cpu_tensor.shape)!s:24s} "
        f"dtype={cpu_tensor.dtype}"
    )

    return report


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    input_names = (
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_grid_thw",
    )

    model_inputs: dict[str, torch.Tensor] = {}

    for name in input_names:
        path = REFERENCE_DIR / f"{name}.npy"

        if not path.is_file():
            raise FileNotFoundError(path)

        model_inputs[name] = torch.from_numpy(
            np.load(path)
        )

    print("===== Load model =====", flush=True)

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

    layer = (
        model.model
        .language_model
        .layers[0]
    )

    print("Layer class:", layer.__class__.__name__)
    print("Load time:", f"{load_seconds:.3f} s")

    captures: dict[str, torch.Tensor] = {}
    captured_call: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    def layer_pre_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del module

        call = {
            key: clone_value(value)
            for key, value in kwargs.items()
        }

        positional_names = (
            "hidden_states",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "use_cache",
            "position_embeddings",
        )

        for index, value in enumerate(args):
            if index >= len(positional_names):
                break

            name = positional_names[index]

            if name not in call:
                call[name] = clone_value(value)

        captured_call.update(call)

        hidden_states = call.get(
            "hidden_states"
        )

        if not isinstance(
            hidden_states,
            torch.Tensor,
        ):
            raise RuntimeError(
                "未捕获到Layer 0 hidden_states。"
            )

        captures["layer_input"] = (
            hidden_states.detach().clone()
        )

        attention_mask = call.get(
            "attention_mask"
        )

        if isinstance(
            attention_mask,
            torch.Tensor,
        ):
            captures["attention_mask"] = (
                attention_mask.detach().clone()
            )

        position_ids = call.get(
            "position_ids"
        )

        if isinstance(
            position_ids,
            torch.Tensor,
        ):
            captures["position_ids"] = (
                position_ids.detach().clone()
            )

        position_embeddings = call.get(
            "position_embeddings"
        )

        if (
            isinstance(
                position_embeddings,
                tuple,
            )
            and len(position_embeddings) == 2
            and isinstance(
                position_embeddings[0],
                torch.Tensor,
            )
            and isinstance(
                position_embeddings[1],
                torch.Tensor,
            )
        ):
            captures["rope_cos"] = (
                position_embeddings[0]
                .detach()
                .clone()
            )

            captures["rope_sin"] = (
                position_embeddings[1]
                .detach()
                .clone()
            )
        else:
            raise RuntimeError(
                "没有捕获到(cos, sin)位置嵌入。"
            )

        past_key_values = call.get(
            "past_key_values"
        )

        metadata["past_key_values_is_none"] = (
            past_key_values is None
        )

        metadata["use_cache"] = call.get(
            "use_cache"
        )

        metadata["call_keyword_names"] = sorted(
            call.keys()
        )

    def layer_output_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        del module, args, kwargs

        value = first_tensor(output)

        if value is None:
            raise RuntimeError(
                "Layer 0输出中没有Tensor。"
            )

        captures["layer_output"] = (
            value.detach().clone()
        )

    def make_output_hook(name: str):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            output: Any,
        ) -> None:
            del module, args

            value = first_tensor(output)

            if value is not None:
                captures[name] = (
                    value.detach().clone()
                )

        return hook

    handles = [
        layer.register_forward_pre_hook(
            layer_pre_hook,
            with_kwargs=True,
        ),
        layer.register_forward_hook(
            layer_output_hook,
            with_kwargs=True,
        ),
    ]

    observed_modules = {
        "input_layernorm_output":
            layer.input_layernorm,

        "q_projection_output":
            layer.self_attn.q_proj,

        "k_projection_output":
            layer.self_attn.k_proj,

        "v_projection_output":
            layer.self_attn.v_proj,

        "query_layernorm_output":
            layer.self_attn.query_layernorm,

        "key_layernorm_output":
            layer.self_attn.key_layernorm,

        "attention_output":
            layer.self_attn,

        "o_projection_output":
            layer.self_attn.o_proj,

        "post_attention_layernorm_output":
            layer.post_attention_layernorm,

        "mlp_gate_output":
            layer.mlp.gate_proj,

        "mlp_up_output":
            layer.mlp.up_proj,

        "mlp_down_output":
            layer.mlp.down_proj,

        "mlp_output":
            layer.mlp,
    }

    for name, module in observed_modules.items():
        handles.append(
            module.register_forward_hook(
                make_output_hook(name)
            )
        )

    print(
        "\n===== Full-model prefill capture =====",
        flush=True,
    )

    forward_start = time.perf_counter()

    with torch.inference_mode():
        model(
            **model_inputs,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )

    forward_seconds = (
        time.perf_counter() - forward_start
    )

    for handle in handles:
        handle.remove()

    print(
        "Full forward time:",
        f"{forward_seconds:.3f} s",
    )

    required = (
        "layer_input",
        "layer_output",
        "attention_mask",
        "rope_cos",
        "rope_sin",
        "input_layernorm_output",
        "q_projection_output",
        "k_projection_output",
        "v_projection_output",
        "query_layernorm_output",
        "key_layernorm_output",
        "attention_output",
        "o_projection_output",
        "post_attention_layernorm_output",
        "mlp_gate_output",
        "mlp_up_output",
        "mlp_down_output",
        "mlp_output",
    )

    missing = [
        name
        for name in required
        if name not in captures
    ]

    if missing:
        raise RuntimeError(
            "缺少捕获张量："
            + ", ".join(missing)
        )

    if not metadata[
        "past_key_values_is_none"
    ]:
        raise RuntimeError(
            "本次Prefill意外使用了KV Cache。"
        )

    print(
        "\n===== Independent Layer 0 call =====",
        flush=True,
    )

    replay_kwargs = {
        key: clone_value(value)
        for key, value in captured_call.items()
    }

    replay_kwargs["hidden_states"] = (
        captures["layer_input"]
    )

    replay_kwargs["attention_mask"] = (
        captures["attention_mask"]
    )

    replay_kwargs["position_embeddings"] = (
        captures["rope_cos"],
        captures["rope_sin"],
    )

    replay_kwargs["past_key_values"] = None
    replay_kwargs["use_cache"] = False

    independent_start = time.perf_counter()

    with torch.inference_mode():
        independent_output = layer(
            **replay_kwargs
        )

    independent_seconds = (
        time.perf_counter()
        - independent_start
    )

    independent_tensor = first_tensor(
        independent_output
    )

    if independent_tensor is None:
        raise RuntimeError(
            "独立Layer 0输出中没有Tensor。"
        )

    independent_tensor = (
        independent_tensor.detach().clone()
    )

    captures[
        "independent_layer_output"
    ] = independent_tensor

    max_error = float(
        (
            independent_tensor
            - captures["layer_output"]
        )
        .abs()
        .max()
        .item()
    )

    mean_error = float(
        (
            independent_tensor
            - captures["layer_output"]
        )
        .abs()
        .mean()
        .item()
    )

    print(
        "Independent time:",
        f"{independent_seconds:.3f} s",
    )
    print(
        "Independent max abs error:",
        max_error,
    )
    print(
        "Independent mean abs error:",
        mean_error,
    )

    if max_error > 1e-7:
        raise RuntimeError(
            "独立Layer 0与完整模型不一致。"
        )

    if tuple(
        captures["layer_input"].shape
    ) != (
        1,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
    ):
        raise RuntimeError(
            "Layer 0输入形状不符合预期："
            f"{captures['layer_input'].shape}"
        )

    if tuple(
        captures["layer_output"].shape
    ) != (
        1,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
    ):
        raise RuntimeError(
            "Layer 0输出形状不符合预期："
            f"{captures['layer_output'].shape}"
        )

    print("\n===== Captured tensors =====")

    tensor_reports: dict[str, Any] = {}

    for name in sorted(captures):
        tensor_reports[name] = save_tensor(
            name,
            captures[name],
        )

    report = {
        "model_revision": (
            PROJECT_DIR
            / "configs/model_revision.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),
        "module_path": (
            "model.language_model.layers.0"
        ),
        "module_class": (
            layer.__class__.__name__
        ),
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": load_seconds,
        "full_forward_seconds": (
            forward_seconds
        ),
        "independent_layer_seconds": (
            independent_seconds
        ),
        "independent_max_abs_error": (
            max_error
        ),
        "independent_mean_abs_error": (
            mean_error
        ),
        "call_metadata": metadata,
        "captured_tensors": tensor_reports,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\nReport:", REPORT_PATH)
    print("Tensor directory:", OUTPUT_DIR)
    print(
        "✅ Decoder Layer 0 Prefill契约捕获成功。"
    )


if __name__ == "__main__":
    main()
