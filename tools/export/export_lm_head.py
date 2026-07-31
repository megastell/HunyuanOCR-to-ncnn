from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

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

SOURCE_INPUT_PATH = (
    PROJECT_DIR
    / "reference/smoke_en_cpu_fp32"
    / "split_contract/lm_head_input.npy"
)

ARTIFACT_DIR = (
    PROJECT_DIR
    / "artifacts/lm_head"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/lm_head_reference.json"
)

EXPECTED_TOKEN = 93892
HIDDEN_SIZE = 1024
VOCAB_SIZE = 120818
TORCH_THREADS = 9


class LMHeadWrapper(nn.Module):
    """只保留 hidden state 到 vocabulary logits 的线性投影。"""

    def __init__(self, lm_head: nn.Module) -> None:
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_state)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


def main() -> None:
    if not SOURCE_INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"找不到 LM Head 输入：{SOURCE_INPUT_PATH}"
        )

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

    source_array = np.load(SOURCE_INPUT_PATH)

    print("原始输入 shape:", source_array.shape)
    print("原始输入 dtype:", source_array.dtype)

    if source_array.shape != (1, 1, HIDDEN_SIZE):
        raise ValueError(
            "LM Head 输入形状与拆分契约不一致："
            f"{source_array.shape}"
        )

    # 从 [1, 1, 1024] 变成更适合独立导出的 [1, 1024]。
    hidden_state = (
        torch.from_numpy(source_array)
        .to(dtype=torch.float32)
        .reshape(1, HIDDEN_SIZE)
        .contiguous()
    )

    print("\n===== 加载模型 =====", flush=True)

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

    model_load_seconds = (
        time.perf_counter() - load_start
    )

    wrapper = LMHeadWrapper(
        model.lm_head
    ).eval()

    print(
        "LM Head weight shape:",
        tuple(wrapper.lm_head.weight.shape),
    )

    if tuple(wrapper.lm_head.weight.shape) != (
        VOCAB_SIZE,
        HIDDEN_SIZE,
    ):
        raise ValueError(
            "LM Head 权重形状与拆分契约不一致。"
        )

    print("\n===== PyTorch 参考推理 =====", flush=True)

    inference_start = time.perf_counter()

    with torch.inference_mode():
        logits = wrapper(hidden_state)

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    if tuple(logits.shape) != (1, VOCAB_SIZE):
        raise ValueError(
            f"输出 shape 错误：{tuple(logits.shape)}"
        )

    predicted_token = int(
        logits.argmax(dim=-1).item()
    )

    top_values, top_indices = torch.topk(
        logits[0],
        k=10,
    )

    top_tokens = [
        {
            "token_id": int(token_id),
            "logit": float(logit),
        }
        for token_id, logit
        in zip(
            top_indices.tolist(),
            top_values.tolist(),
            strict=True,
        )
    ]

    print("输出 shape:", tuple(logits.shape))
    print("Argmax token:", predicted_token)
    print("Expected token:", EXPECTED_TOKEN)
    print("Top 10:")

    for item in top_tokens:
        print(
            f"  token={item['token_id']:6d} "
            f"logit={item['logit']:.9f}"
        )

    if predicted_token != EXPECTED_TOKEN:
        raise RuntimeError(
            "独立 LM Head 的 argmax "
            "与完整模型首 token 不一致。"
        )

    input_raw_path = (
        ARTIFACT_DIR
        / "lm_head_input_f32.bin"
    )

    logits_raw_path = (
        ARTIFACT_DIR
        / "lm_head_logits_f32.bin"
    )

    hidden_state.numpy().astype(
        np.float32,
        copy=False,
    ).tofile(input_raw_path)

    logits.numpy().astype(
        np.float32,
        copy=False,
    ).tofile(logits_raw_path)

    print("\n===== 导出 TorchScript =====", flush=True)

    torchscript_path = (
        ARTIFACT_DIR
        / "lm_head.pt"
    )

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            hidden_state,
            check_trace=True,
            strict=True,
        )

    traced.save(str(torchscript_path))

    # 再次验证 TorchScript 输出。
    loaded_script = torch.jit.load(
        str(torchscript_path),
        map_location="cpu",
    ).eval()

    with torch.inference_mode():
        scripted_logits = loaded_script(
            hidden_state
        )

    script_max_error = float(
        (
            scripted_logits - logits
        )
        .abs()
        .max()
        .item()
    )

    scripted_token = int(
        scripted_logits.argmax(
            dim=-1
        ).item()
    )

    print("TorchScript token:", scripted_token)
    print(
        "TorchScript max abs error:",
        script_max_error,
    )

    if scripted_token != EXPECTED_TOKEN:
        raise RuntimeError(
            "TorchScript 的 argmax 不一致。"
        )

    if script_max_error > 1e-6:
        raise RuntimeError(
            "TorchScript 与 PyTorch 误差过大。"
        )

    report = {
        "model_revision": (
            PROJECT_DIR
            / "configs/model_revision.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),
        "source_input": str(SOURCE_INPUT_PATH),
        "source_input_sha256": sha256_file(
            SOURCE_INPUT_PATH
        ),
        "input_shape": list(hidden_state.shape),
        "input_dtype": str(hidden_state.dtype),
        "output_shape": list(logits.shape),
        "output_dtype": str(logits.dtype),
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "expected_token": EXPECTED_TOKEN,
        "pytorch_argmax_token": predicted_token,
        "torchscript_argmax_token": scripted_token,
        "torchscript_max_abs_error": script_max_error,
        "top_10": top_tokens,
        "logits_min": float(logits.min()),
        "logits_max": float(logits.max()),
        "logits_mean": float(logits.mean()),
        "model_load_seconds": model_load_seconds,
        "pytorch_inference_seconds": inference_seconds,
        "torchscript_path": str(torchscript_path),
        "torchscript_sha256": sha256_file(
            torchscript_path
        ),
        "input_raw_path": str(input_raw_path),
        "input_raw_sha256": sha256_file(
            input_raw_path
        ),
        "logits_raw_path": str(logits_raw_path),
        "logits_raw_sha256": sha256_file(
            logits_raw_path
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": (
            transformers.__version__
        ),
        "device": "cpu",
        "dtype": "float32",
        "torch_threads": TORCH_THREADS,
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
    print("Raw input:", input_raw_path)
    print("Raw logits:", logits_raw_path)
    print("Report:", REPORT_PATH)
    print(
        "✅ LM Head PyTorch/TorchScript "
        "参考结果生成成功。"
    )


if __name__ == "__main__":
    main()
