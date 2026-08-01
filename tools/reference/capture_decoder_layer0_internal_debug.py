from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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

REFERENCE_ROOT = (
    PROJECT_DIR
    / "reference/smoke_en_cpu_fp32"
)

LAYER_REFERENCE_DIR = (
    REFERENCE_ROOT
    / "decoder_layer0_prefill"
)

RAW_OUTPUT_DIR = (
    PROJECT_DIR
    / "artifacts/decoder_layer0_prefill"
    / "debug_refs"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/decoder_layer0_internal_debug_refs.json"
)

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


def load_model_input(name: str) -> torch.Tensor:
    path = REFERENCE_ROOT / f"{name}.npy"

    if not path.is_file():
        raise FileNotFoundError(path)

    return torch.from_numpy(
        np.load(path)
    )


def load_reference(name: str) -> torch.Tensor:
    path = LAYER_REFERENCE_DIR / f"{name}.npy"

    if not path.is_file():
        raise FileNotFoundError(path)

    return torch.from_numpy(
        np.load(path)
    ).float()


def save_capture(
    name: str,
    tensor: torch.Tensor,
) -> dict[str, Any]:
    value = (
        tensor.detach()
        .cpu()
        .float()
        .contiguous()
    )

    npy_path = (
        LAYER_REFERENCE_DIR
        / f"{name}.npy"
    )

    raw_path = (
        RAW_OUTPUT_DIR
        / f"{name}_f32.bin"
    )

    array = value.numpy()

    np.save(npy_path, array)
    array.tofile(raw_path)

    print(
        f"{name:40s} "
        f"shape={tuple(value.shape)!s:24s} "
        f"min={float(value.min()): .7f} "
        f"max={float(value.max()): .7f}"
    )

    return {
        "shape": list(value.shape),
        "dtype": "float32",
        "numel": value.numel(),
        "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
        "npy_sha256": sha256_file(npy_path),
        "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": sha256_file(raw_path),
        "minimum": float(value.min().item()),
        "maximum": float(value.max().item()),
        "mean": float(value.mean().item()),
        "has_nan": bool(
            torch.isnan(value).any().item()
        ),
        "has_inf": bool(
            torch.isinf(value).any().item()
        ),
    }


def error_statistics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise RuntimeError(
            "比较张量形状不一致："
            f"{tuple(actual.shape)} 与 "
            f"{tuple(expected.shape)}"
        )

    difference = (
        actual.float() - expected.float()
    ).abs()

    return {
        "maximum_abs_error": float(
            difference.max().item()
        ),
        "mean_abs_error": float(
            difference.mean().item()
        ),
    }


def main() -> None:
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    LAYER_REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_names = (
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_grid_thw",
    )

    model_inputs = {
        name: load_model_input(name)
        for name in input_names
    }

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

    captures: dict[str, torch.Tensor] = {}

    def make_pre_hook(name: str):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
        ) -> None:
            del module

            if not args:
                raise RuntimeError(
                    f"{name} pre-hook 没有位置参数。"
                )

            value = args[0]

            if not isinstance(
                value,
                torch.Tensor,
            ):
                raise RuntimeError(
                    f"{name} 的第一个输入不是Tensor。"
                )

            captures[name] = (
                value.detach().clone()
            )

        return hook

    handles = [
        layer.self_attn.query_layernorm
        .register_forward_pre_hook(
            make_pre_hook(
                "query_layernorm_input"
            )
        ),

        layer.self_attn.key_layernorm
        .register_forward_pre_hook(
            make_pre_hook(
                "key_layernorm_input"
            )
        ),

        layer.self_attn.o_proj
        .register_forward_pre_hook(
            make_pre_hook(
                "o_projection_input"
            )
        ),

        layer.post_attention_layernorm
        .register_forward_pre_hook(
            make_pre_hook(
                "post_attention_layernorm_input"
            )
        ),

        layer.mlp.down_proj
        .register_forward_pre_hook(
            make_pre_hook(
                "mlp_down_input"
            )
        ),
    ]

    print(
        "===== Full-model capture =====",
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

    expected_shapes = {
        "query_layernorm_input":
            (1, 16, 313, 128),

        "key_layernorm_input":
            (1, 8, 313, 128),

        "o_projection_input":
            (1, 313, 2048),

        "post_attention_layernorm_input":
            (1, 313, 1024),

        "mlp_down_input":
            (1, 313, 3584),
    }

    missing = [
        name
        for name in expected_shapes
        if name not in captures
    ]

    if missing:
        raise RuntimeError(
            "缺少捕获张量："
            + ", ".join(missing)
        )

    for name, expected_shape in (
        expected_shapes.items()
    ):
        actual_shape = tuple(
            captures[name].shape
        )

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{name}形状错误："
                f"实际={actual_shape}，"
                f"预期={expected_shape}"
            )

    print()
    print("===== Captured tensors =====")

    capture_reports: dict[str, Any] = {}

    for name in expected_shapes:
        capture_reports[name] = save_capture(
            name,
            captures[name],
        )

    print()
    print("===== Derived relationship checks =====")

    layer_input = load_reference(
        "layer_input"
    )

    o_projection_output = load_reference(
        "o_projection_output"
    )

    expected_attention_residual = (
        layer_input
        + o_projection_output
    )

    attention_residual_error = (
        error_statistics(
            captures[
                "post_attention_layernorm_input"
            ],
            expected_attention_residual,
        )
    )

    print(
        "Attention residual max error:",
        attention_residual_error[
            "maximum_abs_error"
        ],
    )

    print(
        "Attention residual mean error:",
        attention_residual_error[
            "mean_abs_error"
        ],
    )

    mlp_gate_output = load_reference(
        "mlp_gate_output"
    )

    mlp_up_output = load_reference(
        "mlp_up_output"
    )

    expected_mlp_product = (
        F.silu(mlp_gate_output)
        * mlp_up_output
    )

    mlp_product_error = error_statistics(
        captures["mlp_down_input"],
        expected_mlp_product,
    )

    print(
        "SwiGLU product max error:",
        mlp_product_error[
            "maximum_abs_error"
        ],
    )

    print(
        "SwiGLU product mean error:",
        mlp_product_error[
            "mean_abs_error"
        ],
    )

    if (
        attention_residual_error[
            "maximum_abs_error"
        ] != 0.0
    ):
        raise RuntimeError(
            "Attention残差关系不是逐元素完全一致。"
        )

    if (
        mlp_product_error[
            "maximum_abs_error"
        ] != 0.0
    ):
        raise RuntimeError(
            "SwiGLU乘积关系不是逐元素完全一致。"
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
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": load_seconds,
        "full_forward_seconds": (
            forward_seconds
        ),
        "captures": capture_reports,
        "derived_checks": {
            "attention_residual": (
                attention_residual_error
            ),
            "swiglu_product": (
                mlp_product_error
            ),
        },
        "ncnn_blob_mapping": {
            "query_layernorm_input": "51",
            "key_layernorm_input": "53",
            "o_projection_input": "63",
            "post_attention_layernorm_input":
                "65",
            "mlp_down_input": "74",
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
    print("Model load:", f"{load_seconds:.3f}s")
    print(
        "Full forward:",
        f"{forward_seconds:.3f}s",
    )
    print("Report:", REPORT_PATH)
    print(
        "✅ Decoder Layer 0内部诊断参考捕获成功。"
    )


if __name__ == "__main__":
    main()
