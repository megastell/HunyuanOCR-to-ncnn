from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import (
    AutoProcessor,
    HunYuanVLForConditionalGeneration,
)


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

IMAGE_PATH = (
    Path(os.environ.get(
        "HUNYUANOCR_SMOKE_IMAGE",
        str(PROJECT_DIR / "tests/assets/ocr_smoke_en.png"),
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
    DOCS_DIR / "reference_smoke_cpu_fp32.json"
)

PROMPT = (
    "请逐行识别图片中的所有文字。"
    "只输出图片中的文字本身，保留换行，不要解释。"
)

EXPECTED_TEXT = "HELLO 2026\nNCNN CPU TEST"

RUN_COUNT = 3
MAX_NEW_TOKENS = 32
TORCH_THREADS = 9


def peak_rss_gib() -> float:
    rss_kib = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    return rss_kib / 1024.0 / 1024.0


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def normalize_text(text: str) -> str:
    lines = [
        line.rstrip()
        for line in text.strip().splitlines()
    ]

    return "\n".join(lines)


def tensor_description(
    tensor: torch.Tensor,
) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }


def save_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:
    cpu_tensor = tensor.detach().cpu()

    # NumPy不能直接保存 bfloat16。
    if cpu_tensor.dtype == torch.bfloat16:
        cpu_tensor = cpu_tensor.float()

    np.save(
        REFERENCE_DIR / f"{name}.npy",
        cpu_tensor.numpy(),
    )


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"模型目录不存在：{MODEL_DIR}"
        )

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"测试图片不存在：{IMAGE_PATH}"
        )

    REFERENCE_DIR.mkdir(
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

    print("===== Configuration =====", flush=True)
    print("Model:", MODEL_DIR, flush=True)
    print("Image:", IMAGE_PATH, flush=True)
    print("Prompt:", PROMPT, flush=True)
    print("Threads:", TORCH_THREADS, flush=True)
    print("Max new tokens:", MAX_NEW_TOKENS, flush=True)
    print("Initial peak RSS:", f"{peak_rss_gib():.3f} GiB", flush=True)

    print("\n===== Load processor =====", flush=True)

    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        backend="pil",
        local_files_only=True,
    )

    print(
        "Processor:",
        processor.__class__.__name__,
        flush=True,
    )

    print("\n===== Load model =====", flush=True)

    model_load_start = time.perf_counter()

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
        time.perf_counter() - model_load_start
    )

    print(
        "Model:",
        model.__class__.__name__,
        flush=True,
    )
    print(
        "Model load time:",
        f"{model_load_seconds:.3f} s",
        flush=True,
    )
    print(
        "Peak RSS after model:",
        f"{peak_rss_gib():.3f} GiB",
        flush=True,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "path": str(IMAGE_PATH),
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        },
    ]

    print("\n===== Preprocess =====", flush=True)

    preprocess_start = time.perf_counter()

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    preprocess_seconds = (
        time.perf_counter() - preprocess_start
    )

    input_summary: dict[str, Any] = {}

    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            input_summary[name] = tensor_description(value)
            save_tensor(name, value)

            print(
                f"{name}: "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}",
                flush=True,
            )

    input_length = int(
        inputs["input_ids"].shape[-1]
    )

    print(
        "Preprocess time:",
        f"{preprocess_seconds:.3f} s",
        flush=True,
    )
    print(
        "Input token length:",
        input_length,
        flush=True,
    )

    print("\n===== Deterministic generation =====", flush=True)

    generated_runs: list[list[int]] = []
    decoded_runs: list[str] = []
    generation_seconds: list[float] = []

    for run_index in range(RUN_COUNT):
        run_start = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.0,
        )

        run_seconds = (
            time.perf_counter() - run_start
        )

        new_token_tensor = outputs[
            0,
            input_length:,
        ].detach().cpu()

        token_ids = [
            int(token)
            for token in new_token_tensor.tolist()
        ]

        decoded_text = processor.decode(
            new_token_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        normalized_text = normalize_text(
            decoded_text
        )

        generated_runs.append(token_ids)
        decoded_runs.append(normalized_text)
        generation_seconds.append(run_seconds)

        print(
            f"Run {run_index + 1}: "
            f"{run_seconds:.3f} s, "
            f"{len(token_ids)} new tokens",
            flush=True,
        )
        print(
            f"Token IDs: {token_ids}",
            flush=True,
        )
        print("Output:", flush=True)
        print(normalized_text, flush=True)
        print(flush=True)

    first_tokens = generated_runs[0]
    first_text = decoded_runs[0]

    tokens_identical = all(
        token_ids == first_tokens
        for token_ids in generated_runs
    )

    texts_identical = all(
        text == first_text
        for text in decoded_runs
    )

    if not tokens_identical:
        raise RuntimeError(
            "三次运行生成的 token ID 不一致。"
        )

    if not texts_identical:
        raise RuntimeError(
            "三次运行解码文本不一致。"
        )

    token_path = (
        REFERENCE_DIR
        / "generated_token_ids.json"
    )

    token_path.write_text(
        json.dumps(
            first_tokens,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    output_path = (
        REFERENCE_DIR
        / "output.txt"
    )

    output_path.write_text(
        first_text + "\n",
        encoding="utf-8",
    )

    exact_expected_match = (
        first_text == EXPECTED_TEXT
    )

    report = {
        "model_dir": str(MODEL_DIR),
        "model_revision": (
            PROJECT_DIR
            / "configs/model_revision.txt"
        ).read_text(encoding="utf-8").strip(),
        "image_path": str(IMAGE_PATH),
        "image_sha256": sha256_file(IMAGE_PATH),
        "prompt": PROMPT,
        "expected_text": EXPECTED_TEXT,
        "generated_text": first_text,
        "exact_expected_match": exact_expected_match,
        "run_count": RUN_COUNT,
        "tokens_identical_across_runs": tokens_identical,
        "texts_identical_across_runs": texts_identical,
        "generated_token_ids": first_tokens,
        "input_length": input_length,
        "new_token_count": len(first_tokens),
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "use_cache": True,
        "repetition_penalty": 1.0,
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": model_load_seconds,
        "preprocess_seconds": preprocess_seconds,
        "generation_seconds": generation_seconds,
        "peak_rss_gib": peak_rss_gib(),
        "input_tensors": input_summary,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "cpu",
        "model_dtype": "float32",
        "attention_implementation": "eager",
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("===== Result =====", flush=True)
    print(
        "Tokens identical across 3 runs:",
        tokens_identical,
        flush=True,
    )
    print(
        "Texts identical across 3 runs:",
        texts_identical,
        flush=True,
    )
    print(
        "Exact expected-text match:",
        exact_expected_match,
        flush=True,
    )
    print(
        "Peak RSS:",
        f"{peak_rss_gib():.3f} GiB",
        flush=True,
    )
    print(
        "Report:",
        REPORT_PATH,
        flush=True,
    )
    print(
        "Reference tensors:",
        REFERENCE_DIR,
        flush=True,
    )
    print(
        "✅ 确定性 CPU OCR 参考基准生成成功。",
        flush=True,
    )


if __name__ == "__main__":
    main()
