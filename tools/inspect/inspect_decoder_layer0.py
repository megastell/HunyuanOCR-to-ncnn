from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from transformers import HunYuanVLForConditionalGeneration


MODEL_DIR = (
    Path.home()
    / "work/hunyuanocr/models/HunyuanOCR-1.5"
)


def safe_signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def parameter_count(
    module: torch.nn.Module,
    *,
    recursive: bool = True,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters(
            recurse=recursive,
        )
    )


def main() -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(9)

    print("===== Load model =====", flush=True)

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

    language_model = model.model.language_model
    layer = language_model.layers[0]

    print("\n===== Language model =====")
    print("class:", language_model.__class__.__name__)
    print("forward:", safe_signature(language_model.forward))
    print("decoder layer count:", len(language_model.layers))

    print("\n===== Decoder Layer 0 =====")
    print("class:", layer.__class__.__name__)
    print("parameters:", f"{parameter_count(layer):,}")
    print("forward:", safe_signature(layer.forward))

    print("\n===== Direct children =====")

    for name, child in layer.named_children():
        print(
            f"{name:28s} | "
            f"{child.__class__.__name__:36s} | "
            f"params={parameter_count(child):,}"
        )
        print(
            " " * 31
            + "forward"
            + safe_signature(child.forward)
        )

    print("\n===== Module tree, depth <= 2 =====")

    for name, module in layer.named_modules():
        if not name:
            continue

        if name.count(".") > 1:
            continue

        print(
            f"{name:40s} | "
            f"{module.__class__.__name__:36s} | "
            f"params={parameter_count(module):,}"
        )

        if not isinstance(
            module,
            (
                torch.nn.Linear,
                torch.nn.Embedding,
                torch.nn.ModuleList,
            ),
        ):
            print(
                " " * 43
                + "forward"
                + safe_signature(module.forward)
            )

    print("\n===== Parameter shapes =====")

    for name, parameter in layer.named_parameters():
        print(
            f"{name:52s} "
            f"shape={tuple(parameter.shape)!s:24s} "
            f"dtype={parameter.dtype}"
        )

    rotary_embedding = language_model.rotary_emb

    print("\n===== Rotary embedding =====")
    print(
        "class:",
        rotary_embedding.__class__.__name__,
    )
    print(
        "forward:",
        safe_signature(rotary_embedding.forward),
    )

    print("\n===== Text config =====")

    config = model.config.text_config

    fields = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
        "rope_scaling",
        "attention_bias",
        "attention_dropout",
        "hidden_act",
    )

    for field in fields:
        print(
            f"{field:28s}: "
            f"{getattr(config, field, None)}"
        )

    print("\n✅ Decoder Layer 0结构检查完成。")


if __name__ == "__main__":
    main()
