from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
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
    / "decoder_layer0_prefill"
)

ARTIFACT_DIR = (
    PROJECT_DIR
    / "artifacts/decoder_layer0_prefill"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/decoder_layer0_prefill_reference.json"
)

SEQUENCE_LENGTH = 313
HIDDEN_SIZE = 1024
ATTENTION_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
ROPE_COMPONENTS = 4
TORCH_THREADS = 9


class DecoderLayer0PrefillWrapper(nn.Module):
    """
    HunyuanOCR Decoder Layer 0 的无Cache Prefill包装器。

    静态输入：
    - hidden_states: [1, 313, 1024]
    - attention_mask: [1, 1, 313, 313]
    - rope_cos: [4, 1, 313, 128]
    - rope_sin: [4, 1, 313, 128]

    静态输出：
    - hidden_states: [1, 313, 1024]
    """

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        output = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(
                rope_cos,
                rope_sin,
            ),
        )

        # 当前Transformers版本返回Tensor。
        # 保留兼容逻辑，防止以后返回tuple。
        if isinstance(output, tuple):
            return output[0]

        return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_f32(
    name: str,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    path = REFERENCE_DIR / f"{name}.npy"

    if not path.is_file():
        raise FileNotFoundError(path)

    array = np.load(path)

    if array.shape != expected_shape:
        raise ValueError(
            f"{name}形状错误："
            f"实际={array.shape}，"
            f"预期={expected_shape}"
        )

    if array.dtype != np.float32:
        raise ValueError(
            f"{name}类型错误：{array.dtype}"
        )

    return (
        torch.from_numpy(array)
        .to(dtype=torch.float32)
        .contiguous()
    )


def save_raw(
    name: str,
    tensor: torch.Tensor,
) -> dict[str, Any]:
    path = ARTIFACT_DIR / f"{name}_f32.bin"

    value = (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(np.float32, copy=False)
    )

    value.tofile(path)

    return {
        "path": str(path),
        "shape": list(tensor.shape),
        "dtype": "float32",
        "numel": tensor.numel(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tensor_statistics(
    tensor: torch.Tensor,
) -> dict[str, Any]:
    value = tensor.detach().float()

    return {
        "shape": list(value.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
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


def main() -> None:
    ARTIFACT_DIR.mkdir(
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

    hidden_states = load_f32(
        "layer_input",
        (
            1,
            SEQUENCE_LENGTH,
            HIDDEN_SIZE,
        ),
    )

    attention_mask = load_f32(
        "attention_mask",
        (
            1,
            1,
            SEQUENCE_LENGTH,
            SEQUENCE_LENGTH,
        ),
    )

    rope_cos = load_f32(
        "rope_cos",
        (
            ROPE_COMPONENTS,
            1,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        ),
    )

    rope_sin = load_f32(
        "rope_sin",
        (
            ROPE_COMPONENTS,
            1,
            SEQUENCE_LENGTH,
            HEAD_DIM,
        ),
    )

    expected_output = load_f32(
        "layer_output",
        (
            1,
            SEQUENCE_LENGTH,
            HIDDEN_SIZE,
        ),
    )

    print("===== Inputs =====")
    print(
        "hidden_states:",
        tuple(hidden_states.shape),
    )
    print(
        "attention_mask:",
        tuple(attention_mask.shape),
    )
    print(
        "rope_cos:",
        tuple(rope_cos.shape),
    )
    print(
        "rope_sin:",
        tuple(rope_sin.shape),
    )
    print(
        "expected_output:",
        tuple(expected_output.shape),
    )

    print("\n===== Load model =====", flush=True)

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

    wrapper = DecoderLayer0PrefillWrapper(
        layer
    ).eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in wrapper.parameters()
    )

    print("Layer class:", layer.__class__.__name__)
    print("Parameter count:", f"{parameter_count:,}")
    print("Load time:", f"{load_seconds:.3f} s")

    if parameter_count != 17_303_808:
        raise RuntimeError(
            "Decoder Layer 0参数量与结构契约不一致。"
        )

    print(
        "\n===== Independent PyTorch wrapper =====",
        flush=True,
    )

    inference_start = time.perf_counter()

    with torch.inference_mode():
        pytorch_output = wrapper(
            hidden_states,
            attention_mask,
            rope_cos,
            rope_sin,
        )

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    if tuple(pytorch_output.shape) != (
        1,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
    ):
        raise RuntimeError(
            "Decoder Layer 0输出形状错误："
            f"{tuple(pytorch_output.shape)}"
        )

    pytorch_error = (
        pytorch_output - expected_output
    ).abs()

    pytorch_max_error = float(
        pytorch_error.max().item()
    )

    pytorch_mean_error = float(
        pytorch_error.mean().item()
    )

    print(
        "PyTorch output:",
        tuple(pytorch_output.shape),
    )
    print(
        "PyTorch max abs error:",
        pytorch_max_error,
    )
    print(
        "PyTorch mean abs error:",
        pytorch_mean_error,
    )
    print(
        "PyTorch inference time:",
        f"{inference_seconds:.6f} s",
    )

    if pytorch_max_error > 1e-7:
        raise RuntimeError(
            "静态Wrapper与捕获参考输出不一致。"
        )

    raw_files = {
        "hidden_states": save_raw(
            "hidden_states",
            hidden_states,
        ),
        "attention_mask": save_raw(
            "attention_mask",
            attention_mask,
        ),
        "rope_cos": save_raw(
            "rope_cos",
            rope_cos,
        ),
        "rope_sin": save_raw(
            "rope_sin",
            rope_sin,
        ),
        "expected_output": save_raw(
            "expected_output",
            pytorch_output,
        ),
    }

    print("\n===== Export TorchScript =====", flush=True)

    torchscript_path = (
        ARTIFACT_DIR
        / "decoder_layer0_prefill.pt"
    )

    trace_start = time.perf_counter()

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            (
                hidden_states,
                attention_mask,
                rope_cos,
                rope_sin,
            ),
            check_trace=True,
            strict=False,
        )

    trace_seconds = (
        time.perf_counter() - trace_start
    )

    traced.save(str(torchscript_path))

    loaded_script = torch.jit.load(
        str(torchscript_path),
        map_location="cpu",
    ).eval()

    print(
        "TorchScript graph inputs:",
        len(list(loaded_script.graph.inputs())),
    )

    script_start = time.perf_counter()

    with torch.inference_mode():
        scripted_output = loaded_script(
            hidden_states,
            attention_mask,
            rope_cos,
            rope_sin,
        )

    script_seconds = (
        time.perf_counter() - script_start
    )

    script_error = (
        scripted_output - pytorch_output
    ).abs()

    script_max_error = float(
        script_error.max().item()
    )

    script_mean_error = float(
        script_error.mean().item()
    )

    print(
        "TorchScript output:",
        tuple(scripted_output.shape),
    )
    print(
        "TorchScript max abs error:",
        script_max_error,
    )
    print(
        "TorchScript mean abs error:",
        script_mean_error,
    )
    print(
        "Trace time:",
        f"{trace_seconds:.3f} s",
    )
    print(
        "TorchScript inference time:",
        f"{script_seconds:.6f} s",
    )

    if script_max_error > 1e-6:
        raise RuntimeError(
            "TorchScript与PyTorch输出误差过大。"
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
        "wrapper_class": (
            wrapper.__class__.__name__
        ),
        "parameter_count": parameter_count,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "attention_heads": ATTENTION_HEADS,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rope_components": ROPE_COMPONENTS,
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": load_seconds,
        "pytorch_inference_seconds": (
            inference_seconds
        ),
        "trace_seconds": trace_seconds,
        "torchscript_inference_seconds": (
            script_seconds
        ),
        "pytorch_max_abs_error": (
            pytorch_max_error
        ),
        "pytorch_mean_abs_error": (
            pytorch_mean_error
        ),
        "torchscript_max_abs_error": (
            script_max_error
        ),
        "torchscript_mean_abs_error": (
            script_mean_error
        ),
        "input_statistics": {
            "hidden_states": tensor_statistics(
                hidden_states
            ),
            "attention_mask": tensor_statistics(
                attention_mask
            ),
            "rope_cos": tensor_statistics(
                rope_cos
            ),
            "rope_sin": tensor_statistics(
                rope_sin
            ),
        },
        "output_statistics": tensor_statistics(
            pytorch_output
        ),
        "raw_files": raw_files,
        "torchscript_path": str(
            torchscript_path
        ),
        "torchscript_bytes": (
            torchscript_path.stat().st_size
        ),
        "torchscript_sha256": sha256_file(
            torchscript_path
        ),
        "python_version": (
            platform.python_version()
        ),
        "torch_version": torch.__version__,
        "transformers_version": (
            transformers.__version__
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\nTorchScript:", torchscript_path)
    print(
        "TorchScript size:",
        f"{torchscript_path.stat().st_size / 1024 / 1024:.2f} MiB",
    )
    print("Report:", REPORT_PATH)
    print(
        "✅ Decoder Layer 0 Prefill "
        "PyTorch/TorchScript导出成功。"
    )


if __name__ == "__main__":
    main()
