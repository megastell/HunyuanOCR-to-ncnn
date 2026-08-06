from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from transformers.utils import logging


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
CASE_PATH = PROJECT_DIR / "tests/assets/dynamic_ocr_cases.json"
EXPECTED_PATH = PROJECT_DIR / "tests/assets/dynamic_ocr_expected.json"
REPORT_PATH = PROJECT_DIR / "docs/dynamic_image_reference.json"
REFERENCE_DIR = PROJECT_DIR / "reference/dynamic_image_cpu_fp32"
RECOVERY_DIR = Path.home() / "hunyuanocr-recovery/phase4c"

PROMPT = (
    "请逐行识别图片中的所有文字。只输出图片中的文字本身，"
    "保留换行，不要解释。"
)
TORCH_THREADS = 9
MAX_NEW_TOKENS = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def save_tensor(directory: Path, name: str, tensor: torch.Tensor) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.float()
    path = directory / f"{name}.npy"
    np.save(path, value.numpy())
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "path": path.relative_to(PROJECT_DIR).as_posix(),
        "sha256": sha256(path),
    }


def main() -> None:
    start = time.perf_counter()
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    logging.disable_progress_bar()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    case_document = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    cases = case_document["cases"]
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR), backend="pil", local_files_only=True
    )
    model = (
        HunYuanVLForConditionalGeneration.from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .eval()
    )

    captured: list[dict[str, Any]] = []
    for case in cases:
        case_start = time.perf_counter()
        image_path = PROJECT_DIR / case["path"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_path)},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            image_outputs = model.model.get_image_features(
                inputs["pixel_values"].float(),
                inputs["image_grid_thw"],
                return_dict=True,
            )
            image_features = image_outputs.pooler_output
            text_embeddings = model.model.get_input_embeddings()(inputs["input_ids"])
            fused_embeddings = text_embeddings.clone()
            image_mask = inputs["input_ids"] == model.config.image_token_id
            fused_embeddings = fused_embeddings.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(fused_embeddings),
                image_features,
            )
            position_ids = model.model.compute_3d_position_ids(
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs["image_grid_thw"],
                attention_mask=inputs["attention_mask"],
                mm_token_type_ids=inputs["mm_token_type_ids"],
            )
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.0,
            )

        new_tokens = generated[0, input_length:].detach().cpu()
        token_ids = [int(value) for value in new_tokens.tolist()]
        text = normalize_text(
            processor.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        grid = [int(value) for value in inputs["image_grid_thw"][0].tolist()]
        expected_image_tokens = (grid[1] // 2) * (grid[2] // 2 + 1) + 2
        actual_image_tokens = int(image_mask.sum().item())
        if actual_image_tokens != expected_image_tokens:
            raise RuntimeError(
                f"{case['name']} image token mismatch: "
                f"{actual_image_tokens} != {expected_image_tokens}"
            )
        if list(image_features.shape) != [1, expected_image_tokens, 1024]:
            raise RuntimeError(
                f"{case['name']} image feature shape: {tuple(image_features.shape)}"
            )
        if position_ids is None:
            raise RuntimeError(f"{case['name']} position IDs are missing")

        tensor_dir = REFERENCE_DIR / case["name"]
        tensors = {
            name: save_tensor(tensor_dir, name, value)
            for name, value in {
                "pixel_values": inputs["pixel_values"].float(),
                "image_grid_thw": inputs["image_grid_thw"],
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "mm_token_type_ids": inputs["mm_token_type_ids"],
                "position_ids": position_ids,
                "image_features": image_features,
                "fused_embeddings": fused_embeddings,
            }.items()
        }
        result = {
            **case,
            "grid_thw": grid,
            "sequence_length": input_length,
            "image_token_count": actual_image_tokens,
            "image_token_span": [
                int(torch.nonzero(image_mask[0], as_tuple=False)[0].item()),
                int(torch.nonzero(image_mask[0], as_tuple=False)[-1].item()) + 1,
            ],
            "generated_token_ids": token_ids,
            "generated_text": text,
            "eos_reached": bool(
                token_ids and token_ids[-1] == processor.tokenizer.eos_token_id
            ),
            "elapsed_seconds": time.perf_counter() - case_start,
            "tensors": tensors,
        }
        captured.append(result)
        print(
            f"{case['name']}: grid={grid} sequence={input_length} "
            f"image_tokens={actual_image_tokens} tokens={token_ids}"
        )
        print(text, flush=True)

    expected = {
        "format": "HUNYUANOCR_DYNAMIC_OCR_EXPECTED_V1",
        "model_revision": (PROJECT_DIR / "configs/model_revision.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "prompt": PROMPT,
        "cases": [
            {
                key: case[key]
                for key in (
                    "name",
                    "path",
                    "sha256",
                    "grid_thw",
                    "sequence_length",
                    "image_token_count",
                    "image_token_span",
                    "generated_token_ids",
                    "generated_text",
                    "eos_reached",
                )
            }
            for case in captured
        ],
    }
    EXPECTED_PATH.write_text(
        json.dumps(expected, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        **expected,
        "torch_threads": TORCH_THREADS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "total_seconds": time.perf_counter() - start,
        "cases": captured,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Expected: {EXPECTED_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
