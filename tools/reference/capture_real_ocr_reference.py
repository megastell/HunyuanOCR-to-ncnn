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
PNG_IMAGE_PATH = PROJECT_DIR / "tests/assets/ocr_receipt_real.png"
JPEG_IMAGE_PATH = PROJECT_DIR / "tests/assets/ocr_receipt_real.jpg"
JPEG_RGB_PATH = PROJECT_DIR / "tests/assets/ocr_receipt_real_stb.ppm"
EXPECTED_PATH = PROJECT_DIR / "tests/assets/real_ocr_expected.json"
REPORT_PATH = PROJECT_DIR / "docs/real_ocr_reference.json"
REFERENCE_DIR = PROJECT_DIR / "reference/real_ocr_cpu_fp32/ocr_receipt_real"
RECOVERY_DIR = Path.home() / "hunyuanocr-recovery/phase4e/reference"

PROMPT = (
    "\u8bf7\u9010\u884c\u8bc6\u522b\u56fe\u7247\u4e2d\u7684\u6240\u6709\u6587\u5b57\u3002\u53ea\u8f93\u51fa\u56fe\u7247"
    "\u4e2d\u7684\u6587\u5b57\u672c\u8eab\uff0c\u4fdd\u7559\u6362\u884c\uff0c\u4e0d\u8981\u89e3\u91ca\u3002"
)
TORCH_THREADS = 9
MAX_NEW_TOKENS = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_tensor(case_name: str, name: str, tensor: torch.Tensor) -> dict[str, Any]:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.float()
    path = REFERENCE_DIR / f"{case_name}_{name}.npy"
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
    case_specs = [
        {
            "name": "ocr_receipt_real_png",
            "runtime_path": PNG_IMAGE_PATH,
            "reference_path": PNG_IMAGE_PATH,
            "contract": "lossless PNG pixels",
        },
        {
            "name": "ocr_receipt_real_jpeg",
            "runtime_path": JPEG_IMAGE_PATH,
            "reference_path": JPEG_RGB_PATH,
            "contract": "production stb_image-decoded RGB PPM",
        },
    ]
    cases = []
    tensors = {}
    for spec in case_specs:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "path": str(spec["reference_path"])},
                {"type": "text", "text": PROMPT},
            ],
        }]
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
        text = processor.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        grid = [int(value) for value in inputs["image_grid_thw"][0].tolist()]
        image_mask = inputs["input_ids"] == model.config.image_token_id
        image_positions = torch.nonzero(image_mask[0], as_tuple=False)
        runtime_path = spec["runtime_path"]
        reference_path = spec["reference_path"]
        cases.append({
            "name": spec["name"],
            "path": runtime_path.relative_to(PROJECT_DIR).as_posix(),
            "sha256": sha256(runtime_path),
            "reference_pixel_path": (
                reference_path.relative_to(PROJECT_DIR).as_posix()
            ),
            "reference_pixel_sha256": sha256(reference_path),
            "pixel_contract": spec["contract"],
            "source": {
                "repository": "https://github.com/Tencent-Hunyuan/HunyuanOCR.git",
                "revision": "c55965d3da1e6f41987abec8068f2e70851318bc",
                "path": "HunyuanOCR_v1.0/assets/ie_parallel.jpg",
            },
            "grid_thw": grid,
            "sequence_length": input_length,
            "image_token_count": int(image_mask.sum().item()),
            "image_token_span": [
                int(image_positions[0].item()),
                int(image_positions[-1].item()) + 1,
            ],
            "generated_token_ids": token_ids,
            "generated_text": text,
            "eos_reached": bool(
                token_ids and token_ids[-1] == processor.tokenizer.eos_token_id
            ),
        })
        tensors[spec["name"]] = {
            name: save_tensor(spec["name"], name, value)
            for name, value in {
                "pixel_values": inputs["pixel_values"].float(),
                "image_grid_thw": inputs["image_grid_thw"],
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "mm_token_type_ids": inputs["mm_token_type_ids"],
                "position_ids": position_ids,
                "image_features": image_features,
            }.items()
        }
        print(
            f"{spec['name']}: grid={grid} sequence={input_length} "
            f"tokens={len(token_ids)}"
        )
    expected = {
        "format": "HUNYUANOCR_REAL_OCR_EXPECTED_V2",
        "model_revision": (PROJECT_DIR / "configs/model_revision.txt")
        .read_text(encoding="utf-8").strip(),
        "prompt": PROMPT,
        "cases": cases,
    }
    EXPECTED_PATH.write_text(
        json.dumps(expected, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        json.dumps({
            **expected,
            "torch_threads": TORCH_THREADS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "elapsed_seconds": time.perf_counter() - start,
            "tensors": tensors,
        }, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Expected: {EXPECTED_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
