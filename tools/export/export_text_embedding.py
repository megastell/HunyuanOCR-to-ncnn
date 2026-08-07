from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
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
PNNX_PATH = Path.home() / "work/hunyuanocr/.venv-pnnx/bin/pnnx"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DOCS_DIR = PROJECT_DIR / "docs"
REFERENCE_ROOT = PROJECT_DIR / "reference/smoke_en_cpu_fp32"

SOURCE_IDS_PATH = (
    REFERENCE_ROOT / "input_ids.npy"
)

CAPTURED_OUTPUT_PATH = (
    REFERENCE_ROOT / "split_contract/text_embedding_output.npy"
)

ARTIFACT_DIR = (
    ARTIFACTS_DIR / "text_embedding"
)

REPORT_PATH = (
    DOCS_DIR / "text_embedding_reference.json"
)

SEQUENCE_LENGTH = 313
HIDDEN_SIZE = 1024
VOCAB_SIZE = 120818
TORCH_THREADS = 9


class TextEmbeddingWrapper(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(input_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export text embedding with pnnx.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--pnnx", type=Path, default=PNNX_PATH)
    parser.add_argument("--skip-pnnx", action="store_true")
    return parser.parse_args()


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
    args = parse_args()
    global MODEL_DIR, PNNX_PATH, ARTIFACTS_DIR, DOCS_DIR
    global REFERENCE_ROOT, SOURCE_IDS_PATH, CAPTURED_OUTPUT_PATH
    global ARTIFACT_DIR, REPORT_PATH
    MODEL_DIR = args.model_dir.resolve()
    PNNX_PATH = args.pnnx.resolve()
    ARTIFACTS_DIR = args.artifacts_dir.resolve()
    DOCS_DIR = args.docs_dir.resolve()
    REFERENCE_ROOT = args.reference_dir.resolve()
    SOURCE_IDS_PATH = REFERENCE_ROOT / "input_ids.npy"
    CAPTURED_OUTPUT_PATH = (
        REFERENCE_ROOT / "split_contract/text_embedding_output.npy"
    )
    ARTIFACT_DIR = ARTIFACTS_DIR / "text_embedding"
    REPORT_PATH = DOCS_DIR / "text_embedding_reference.json"

    if not SOURCE_IDS_PATH.is_file():
        raise FileNotFoundError(SOURCE_IDS_PATH)

    if not CAPTURED_OUTPUT_PATH.is_file():
        raise FileNotFoundError(CAPTURED_OUTPUT_PATH)

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

    source_ids = np.load(SOURCE_IDS_PATH)
    captured_output = np.load(
        CAPTURED_OUTPUT_PATH
    )

    print("原始 input_ids shape:", source_ids.shape)
    print("原始 input_ids dtype:", source_ids.dtype)
    print(
        "捕获输出 shape:",
        captured_output.shape,
    )

    if source_ids.shape != (
        1,
        SEQUENCE_LENGTH,
    ):
        raise ValueError(
            f"input_ids shape 不一致：{source_ids.shape}"
        )

    if captured_output.shape != (
        1,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
    ):
        raise ValueError(
            "text_embedding_output shape 不一致："
            f"{captured_output.shape}"
        )

    input_ids = (
        torch.from_numpy(source_ids)
        .to(dtype=torch.int64)
        .reshape(SEQUENCE_LENGTH)
        .contiguous()
    )

    reference_output = (
        torch.from_numpy(captured_output)
        .to(dtype=torch.float32)
        .reshape(
            SEQUENCE_LENGTH,
            HIDDEN_SIZE,
        )
        .contiguous()
    )

    minimum_id = int(input_ids.min())
    maximum_id = int(input_ids.max())

    print("Token ID min:", minimum_id)
    print("Token ID max:", maximum_id)

    if minimum_id < 0 or maximum_id >= VOCAB_SIZE:
        raise ValueError(
            "发现超出词表范围的 token ID。"
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

    load_seconds = (
        time.perf_counter() - load_start
    )

    embedding = (
        model.model
        .language_model
        .embed_tokens
    )

    wrapper = TextEmbeddingWrapper(
        embedding
    ).eval()

    print(
        "Embedding weight shape:",
        tuple(embedding.weight.shape),
    )

    if tuple(embedding.weight.shape) != (
        VOCAB_SIZE,
        HIDDEN_SIZE,
    ):
        raise ValueError(
            "Embedding 权重形状不符合拆分契约。"
        )

    print(
        "\n===== PyTorch参考推理 =====",
        flush=True,
    )

    inference_start = time.perf_counter()

    with torch.inference_mode():
        output = wrapper(input_ids)

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    if tuple(output.shape) != (
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
    ):
        raise ValueError(
            f"Embedding输出形状错误：{output.shape}"
        )

    captured_max_error = float(
        (
            output - reference_output
        )
        .abs()
        .max()
        .item()
    )

    captured_mean_error = float(
        (
            output - reference_output
        )
        .abs()
        .mean()
        .item()
    )

    print("输出 shape:", tuple(output.shape))
    print(
        "与完整模型捕获结果最大误差:",
        captured_max_error,
    )
    print(
        "与完整模型捕获结果平均误差:",
        captured_mean_error,
    )

    if captured_max_error > 1e-7:
        raise RuntimeError(
            "独立Embedding与完整模型捕获结果不一致。"
        )

    input_raw_path = (
        ARTIFACT_DIR
        / "input_ids_i32.bin"
    )

    output_raw_path = (
        ARTIFACT_DIR
        / "text_embedding_output_f32.bin"
    )

    input_ids.numpy().astype(
        np.int32,
        copy=False,
    ).tofile(input_raw_path)

    output.numpy().astype(
        np.float32,
        copy=False,
    ).tofile(output_raw_path)

    print("\n===== 导出TorchScript =====")

    torchscript_path = (
        ARTIFACT_DIR
        / "text_embedding.pt"
    )

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            input_ids,
            check_trace=True,
            strict=True,
        )

    traced.save(str(torchscript_path))

    loaded_script = torch.jit.load(
        str(torchscript_path),
        map_location="cpu",
    ).eval()

    with torch.inference_mode():
        scripted_output = loaded_script(
            input_ids
        )

    script_max_error = float(
        (
            scripted_output - output
        )
        .abs()
        .max()
        .item()
    )

    print(
        "TorchScript输出 shape:",
        tuple(scripted_output.shape),
    )
    print(
        "TorchScript最大绝对误差:",
        script_max_error,
    )

    if script_max_error > 1e-7:
        raise RuntimeError(
            "TorchScript与PyTorch输出不一致。"
        )

    pnnx_command: list[str] | None = None
    pnnx_log_path = DOCS_DIR / "text_embedding_pnnx_fp32.txt"
    if not args.skip_pnnx:
        pnnx_command = [
            str(PNNX_PATH),
            str(torchscript_path),
            "inputshape=[313]i64",
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
        ).read_text(
            encoding="utf-8"
        ).strip(),
        "source_ids": str(SOURCE_IDS_PATH),
        "source_ids_sha256": sha256_file(
            SOURCE_IDS_PATH
        ),
        "captured_output": str(
            CAPTURED_OUTPUT_PATH
        ),
        "captured_output_sha256": sha256_file(
            CAPTURED_OUTPUT_PATH
        ),
        "input_shape": list(input_ids.shape),
        "pytorch_input_dtype": str(
            input_ids.dtype
        ),
        "ncnn_input_dtype": "int32",
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "minimum_token_id": minimum_id,
        "maximum_token_id": maximum_id,
        "captured_max_abs_error": (
            captured_max_error
        ),
        "captured_mean_abs_error": (
            captured_mean_error
        ),
        "torchscript_max_abs_error": (
            script_max_error
        ),
        "model_load_seconds": load_seconds,
        "pytorch_inference_seconds": (
            inference_seconds
        ),
        "torchscript_path": str(
            torchscript_path
        ),
        "torchscript_sha256": sha256_file(
            torchscript_path
        ),
        "pnnx_command": pnnx_command,
        "pnnx_log": str(pnnx_log_path) if pnnx_command is not None else None,
        "input_raw_path": str(input_raw_path),
        "input_raw_sha256": sha256_file(
            input_raw_path
        ),
        "output_raw_path": str(
            output_raw_path
        ),
        "output_raw_sha256": sha256_file(
            output_raw_path
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
    print("ncnn int32输入:", input_raw_path)
    print("FP32参考输出:", output_raw_path)
    print("报告:", REPORT_PATH)
    print(
        "✅ Text Embedding PyTorch/"
        "TorchScript参考结果生成成功。"
    )


if __name__ == "__main__":
    main()
