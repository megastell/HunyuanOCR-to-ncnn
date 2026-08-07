from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

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
PNNX_PATH = Path.home() / "work/hunyuanocr/.venv-pnnx/bin/pnnx"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DOCS_DIR = PROJECT_DIR / "docs"

REFERENCE_DIR = (
    PROJECT_DIR / "reference/smoke_en_cpu_fp32"
)

SPLIT_DIR = REFERENCE_DIR / "split_contract"
ARTIFACT_DIR = ARTIFACTS_DIR / "final_norm"
REPORT_PATH = DOCS_DIR / "final_norm_reference.json"

SEQUENCE_LENGTH = 313
HIDDEN_SIZE = 1024
EPSILON = 1e-5
TORCH_THREADS = 9


class FinalNormWrapper(nn.Module):
    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final RMSNorm with pnnx.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--pnnx", type=Path, default=PNNX_PATH)
    parser.add_argument("--skip-pnnx", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    global MODEL_DIR, PNNX_PATH, ARTIFACTS_DIR, DOCS_DIR
    global REFERENCE_DIR, SPLIT_DIR, ARTIFACT_DIR, REPORT_PATH
    MODEL_DIR = args.model_dir.resolve()
    PNNX_PATH = args.pnnx.resolve()
    ARTIFACTS_DIR = args.artifacts_dir.resolve()
    DOCS_DIR = args.docs_dir.resolve()
    REFERENCE_DIR = args.reference_dir.resolve()
    SPLIT_DIR = REFERENCE_DIR / "split_contract"
    ARTIFACT_DIR = ARTIFACTS_DIR / "final_norm"
    REPORT_PATH = DOCS_DIR / "final_norm_reference.json"

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

    print("===== 加载完整模型 =====", flush=True)

    start = time.perf_counter()

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

    load_seconds = time.perf_counter() - start

    final_norm = model.model.language_model.norm

    print("Norm class:", final_norm.__class__.__name__)
    print("Weight shape:", tuple(final_norm.weight.shape))
    print("Configured epsilon:", model.config.text_config.rms_norm_eps)

    if tuple(final_norm.weight.shape) != (HIDDEN_SIZE,):
        raise RuntimeError("Final RMSNorm 权重形状不正确。")

    captures: dict[str, torch.Tensor] = {}

    def pre_hook(
        module: nn.Module,
        args: tuple[object, ...],
    ) -> None:
        del module

        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("没有捕获到Final RMSNorm输入。")

        captures["input"] = args[0].detach().clone()

    def output_hook(
        module: nn.Module,
        args: tuple[object, ...],
        output: object,
    ) -> None:
        del module, args

        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Final RMSNorm输出不是Tensor。")

        captures["output"] = output.detach().clone()

    pre_handle = final_norm.register_forward_pre_hook(pre_hook)
    output_handle = final_norm.register_forward_hook(output_hook)

    print("\n===== 捕获完整模型中的Final RMSNorm =====", flush=True)

    forward_start = time.perf_counter()

    with torch.inference_mode():
        model(
            **model_inputs,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )

    forward_seconds = time.perf_counter() - forward_start

    pre_handle.remove()
    output_handle.remove()

    if "input" not in captures or "output" not in captures:
        raise RuntimeError("Final RMSNorm捕获失败。")

    captured_input = (
        captures["input"]
        .reshape(SEQUENCE_LENGTH, HIDDEN_SIZE)
        .contiguous()
    )

    captured_output = (
        captures["output"]
        .reshape(SEQUENCE_LENGTH, HIDDEN_SIZE)
        .contiguous()
    )

    print("Captured input:", tuple(captured_input.shape))
    print("Captured output:", tuple(captured_output.shape))

    existing_output_path = SPLIT_DIR / "final_norm_output.npy"
    existing_output = torch.from_numpy(
        np.load(existing_output_path)
    ).reshape(SEQUENCE_LENGTH, HIDDEN_SIZE)

    previous_capture_error = float(
        (captured_output - existing_output)
        .abs()
        .max()
        .item()
    )

    print(
        "与上一阶段捕获结果最大误差:",
        previous_capture_error,
    )

    wrapper = FinalNormWrapper(final_norm).eval()

    print("\n===== 独立PyTorch Final RMSNorm =====", flush=True)

    with torch.inference_mode():
        direct_output = wrapper(captured_input)

    direct_max_error = float(
        (direct_output - captured_output)
        .abs()
        .max()
        .item()
    )

    direct_mean_error = float(
        (direct_output - captured_output)
        .abs()
        .mean()
        .item()
    )

    print("Direct output:", tuple(direct_output.shape))
    print("独立模块最大误差:", direct_max_error)
    print("独立模块平均误差:", direct_mean_error)

    if direct_max_error > 1e-7:
        raise RuntimeError("独立Final RMSNorm与完整模型不一致。")

    input_raw_path = ARTIFACT_DIR / "final_norm_input_f32.bin"
    output_raw_path = ARTIFACT_DIR / "final_norm_output_f32.bin"

    captured_input.numpy().astype(
        np.float32,
        copy=False,
    ).tofile(input_raw_path)

    direct_output.numpy().astype(
        np.float32,
        copy=False,
    ).tofile(output_raw_path)

    print("\n===== 导出TorchScript =====", flush=True)

    torchscript_path = ARTIFACT_DIR / "final_norm.pt"

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            captured_input,
            check_trace=True,
            strict=True,
        )

    traced.save(str(torchscript_path))

    loaded_script = torch.jit.load(
        str(torchscript_path),
        map_location="cpu",
    ).eval()

    with torch.inference_mode():
        scripted_output = loaded_script(captured_input)

    script_max_error = float(
        (scripted_output - direct_output)
        .abs()
        .max()
        .item()
    )

    print("TorchScript输出:", tuple(scripted_output.shape))
    print("TorchScript最大误差:", script_max_error)

    if script_max_error > 1e-7:
        raise RuntimeError("TorchScript输出不一致。")

    pnnx_command: list[str] | None = None
    pnnx_log_path = DOCS_DIR / "final_norm_pnnx_fp32.txt"
    if not args.skip_pnnx:
        pnnx_command = [
            str(PNNX_PATH),
            str(torchscript_path),
            "inputshape=[313,1024]f32",
            "fp16=0",
            "optlevel=2",
            "device=cpu",
        ]
        conversion = subprocess.run(
            pnnx_command,
            cwd=ARTIFACT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        pnnx_log_path.write_text(conversion.stdout, encoding="utf-8")
        if conversion.returncode != 0:
            raise RuntimeError(
                f"pnnx failed with status {conversion.returncode}; "
                f"see {pnnx_log_path}"
            )

    report = {
        "model_revision": (
            PROJECT_DIR
            / "configs/model_revision.txt"
        ).read_text(encoding="utf-8").strip(),
        "module_path": "model.language_model.norm",
        "module_class": final_norm.__class__.__name__,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "epsilon": model.config.text_config.rms_norm_eps,
        "input_shape": list(captured_input.shape),
        "output_shape": list(direct_output.shape),
        "dtype": "float32",
        "previous_capture_max_abs_error": previous_capture_error,
        "direct_max_abs_error": direct_max_error,
        "direct_mean_abs_error": direct_mean_error,
        "torchscript_max_abs_error": script_max_error,
        "model_load_seconds": load_seconds,
        "full_forward_seconds": forward_seconds,
        "torchscript_path": str(torchscript_path),
        "torchscript_sha256": sha256_file(torchscript_path),
        "pnnx_command": pnnx_command,
        "pnnx_log": str(pnnx_log_path) if pnnx_command is not None else None,
        "input_raw_path": str(input_raw_path),
        "input_raw_sha256": sha256_file(input_raw_path),
        "output_raw_path": str(output_raw_path),
        "output_raw_sha256": sha256_file(output_raw_path),
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
    print("Input:", input_raw_path)
    print("Output:", output_raw_path)
    print("Report:", REPORT_PATH)
    print("✅ Final RMSNorm PyTorch参考导出成功。")


if __name__ == "__main__":
    main()
