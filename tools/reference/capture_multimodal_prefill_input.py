from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from transformers.models.hunyuan_vl.image_processing_pil_hunyuan_vl import (
    smart_resize,
)
from transformers.utils import logging


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
IMAGE_PATH = PROJECT_DIR / "tests/assets/ocr_smoke_en.png"
BASE_SCRIPT = PROJECT_DIR / "tools/reference/run_reference_smoke.py"
REFERENCE_NAME = "multimodal_prefill_input"
REFERENCE_DIR = PROJECT_DIR / "reference/smoke_en_cpu_fp32" / REFERENCE_NAME
RAW_DIR = PROJECT_DIR / "artifacts" / REFERENCE_NAME / "reference"
REPORT_PATH = PROJECT_DIR / "docs/multimodal_prefill_input_reference.json"

TORCH_THREADS = 9
PATCH_SIZE = 16
MERGE_SIZE = 2
EXPECTED_SEQUENCE_LENGTH = 313
EXPECTED_IMAGE_TOKENS = 288
EXPECTED_GRID = [1, 22, 50]


def load_base_module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "hunyuanocr_reference_smoke", BASE_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().cpu()
    numeric = detached.float()
    finite = numeric[torch.isfinite(numeric)]
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "minimum_finite": float(finite.min().item()),
        "maximum_finite": float(finite.max().item()),
        "mean_finite": float(finite.mean().item()),
    }


def save_tensor(name: str, value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().cpu().contiguous()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    npy_path = REFERENCE_DIR / f"{name}.npy"
    array = detached.numpy()
    np.save(npy_path, array)

    tags = {
        torch.float32: "f32",
        torch.int64: "i64",
        torch.uint8: "u8",
        torch.bool: "bool",
    }
    tag = tags.get(detached.dtype)
    if tag is None:
        raise RuntimeError(f"Unsupported dtype for {name}: {detached.dtype}")
    raw_path = RAW_DIR / f"{name}_{tag}.bin"
    array.tofile(raw_path)

    report = tensor_summary(detached)
    report.update(
        {
            "npy_path": npy_path.relative_to(PROJECT_DIR).as_posix(),
            "raw_path": raw_path.relative_to(PROJECT_DIR).as_posix(),
            "raw_bytes": raw_path.stat().st_size,
        }
    )
    return report


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"Shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    return float((actual.float() - expected.float()).abs().max().item())


def contiguous_spans(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    spans: list[list[int]] = []
    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            spans.append([start, previous + 1])
            start = index
        previous = index
    spans.append([start, previous + 1])
    return spans


def main() -> None:
    start = time.perf_counter()
    logging.disable_progress_bar()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    base = load_base_module()
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR), backend="pil", local_files_only=True
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(IMAGE_PATH)},
                {"type": "text", "text": base.PROMPT},
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

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    mm_token_type_ids = inputs["mm_token_type_ids"]
    pixel_values = inputs["pixel_values"].float()
    image_grid_thw = inputs["image_grid_thw"]

    if list(input_ids.shape) != [1, EXPECTED_SEQUENCE_LENGTH]:
        raise RuntimeError(f"Unexpected input_ids shape: {tuple(input_ids.shape)}")
    if image_grid_thw[0].tolist() != EXPECTED_GRID:
        raise RuntimeError(f"Unexpected image grid: {image_grid_thw.tolist()}")

    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")
        original_width, original_height = image.size
        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
            factor=PATCH_SIZE * MERGE_SIZE,
            min_pixels=processor.image_processor.size["shortest_edge"],
            max_pixels=processor.image_processor.size["longest_edge"],
        )
        resized = image.resize((resized_width, resized_height))
        original_rgb = np.asarray(image, dtype=np.uint8).copy()
        resized_rgb = np.asarray(resized, dtype=np.uint8).copy()

    normalized = resized_rgb.astype(np.float32).transpose(2, 0, 1)
    normalized *= np.float32(processor.image_processor.rescale_factor)
    mean = np.asarray(processor.image_processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_processor.image_std, dtype=np.float32)
    normalized = (normalized - mean[:, None, None]) / std[:, None, None]

    grid_h = resized_height // PATCH_SIZE
    grid_w = resized_width // PATCH_SIZE
    manual_pixel_values = (
        normalized.reshape(
            1,
            1,
            1,
            3,
            grid_h // MERGE_SIZE,
            MERGE_SIZE,
            PATCH_SIZE,
            grid_w // MERGE_SIZE,
            MERGE_SIZE,
            PATCH_SIZE,
        )
        .transpose(0, 1, 4, 5, 7, 8, 3, 2, 6, 9)
        .reshape(grid_h * grid_w, 3 * PATCH_SIZE * PATCH_SIZE)
    )
    manual_pixel_tensor = torch.from_numpy(manual_pixel_values.copy())
    manual_pixel_error = max_abs_error(manual_pixel_tensor, pixel_values)
    if manual_pixel_error > 1.0e-6:
        raise RuntimeError(
            f"Manual processor reconstruction differs: {manual_pixel_error}"
        )

    model_load_start = time.perf_counter()
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
    model_load_seconds = time.perf_counter() - model_load_start

    with torch.inference_mode():
        text_embeddings = model.model.get_input_embeddings()(input_ids)
        image_features = model.model.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        ).pooler_output
        fused_embeddings = text_embeddings.clone()
        image_mask = input_ids == model.config.image_token_id
        expanded_mask = image_mask.unsqueeze(-1).expand_as(fused_embeddings)
        fused_embeddings = fused_embeddings.masked_scatter(
            expanded_mask, image_features
        )
        position_ids = model.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )

    if image_features.shape != (1, EXPECTED_IMAGE_TOKENS, 1024):
        raise RuntimeError(f"Unexpected image features: {image_features.shape}")
    if int(image_mask.sum().item()) != EXPECTED_IMAGE_TOKENS:
        raise RuntimeError(f"Unexpected image token count: {image_mask.sum()}")
    if position_ids is None:
        raise RuntimeError("Multimodal position ids were not generated")

    expected_fused = torch.from_numpy(
        np.load(
            PROJECT_DIR
            / "reference/smoke_en_cpu_fp32/decoder_layer0_prefill_kv"
            / "layer0_hidden_states.npy"
        )
    ).float()
    fused_error = max_abs_error(fused_embeddings, expected_fused)
    if fused_error != 0.0:
        raise RuntimeError(f"Fused prefill boundary differs: {fused_error}")

    original_rgb_tensor = torch.from_numpy(original_rgb)
    resized_rgb_tensor = torch.from_numpy(resized_rgb)
    tensor_reports = {
        "original_rgb": save_tensor("original_rgb", original_rgb_tensor),
        "resized_rgb": save_tensor("resized_rgb", resized_rgb_tensor),
        "pixel_values": save_tensor("pixel_values", pixel_values),
        "image_grid_thw": save_tensor("image_grid_thw", image_grid_thw),
        "input_ids": save_tensor("input_ids", input_ids),
        "attention_mask": save_tensor("attention_mask", attention_mask),
        "mm_token_type_ids": save_tensor(
            "mm_token_type_ids", mm_token_type_ids
        ),
        "image_mask": save_tensor("image_mask", image_mask.to(torch.uint8)),
        "text_embeddings": save_tensor("text_embeddings", text_embeddings),
        "image_features": save_tensor("image_features", image_features),
        "fused_embeddings": save_tensor("fused_embeddings", fused_embeddings),
        "position_ids": save_tensor("position_ids", position_ids),
    }

    ids = input_ids[0].tolist()
    image_indices = torch.nonzero(image_mask[0], as_tuple=False).flatten().tolist()
    mm_indices = torch.nonzero(
        mm_token_type_ids[0] == 1, as_tuple=False
    ).flatten().tolist()
    unique_ids: list[int] = []
    for token_id in ids:
        if token_id not in unique_ids:
            unique_ids.append(token_id)

    report = {
        "model_revision": (
            PROJECT_DIR / "configs/model_revision.txt"
        ).read_text(encoding="utf-8").strip(),
        "image_path": IMAGE_PATH.relative_to(PROJECT_DIR).as_posix(),
        "image_sha256": sha256_file(IMAGE_PATH),
        "prompt": base.PROMPT,
        "processor_class": processor.__class__.__name__,
        "image_processor_class": processor.image_processor.__class__.__name__,
        "original_size": [original_height, original_width],
        "resized_size": [resized_height, resized_width],
        "smart_resize_factor": PATCH_SIZE * MERGE_SIZE,
        "min_pixels": processor.image_processor.size["shortest_edge"],
        "max_pixels": processor.image_processor.size["longest_edge"],
        "rescale_factor": processor.image_processor.rescale_factor,
        "image_mean": processor.image_processor.image_mean,
        "image_std": processor.image_processor.image_std,
        "patch_size": PATCH_SIZE,
        "merge_size": MERGE_SIZE,
        "grid_thw": image_grid_thw.tolist(),
        "sequence_length": len(ids),
        "input_ids": ids,
        "unique_input_ids": unique_ids,
        "image_token_id": int(model.config.image_token_id),
        "image_token_count": len(image_indices),
        "image_token_spans": contiguous_spans(image_indices),
        "mm_type_one_count": len(mm_indices),
        "mm_type_one_spans": contiguous_spans(mm_indices),
        "prefix_input_ids": ids[: image_indices[0]],
        "suffix_input_ids": ids[image_indices[-1] + 1 :],
        "manual_pixel_values_max_abs_error": manual_pixel_error,
        "fused_prefill_hidden_max_abs_error": fused_error,
        "model_load_seconds": model_load_seconds,
        "capture_seconds": time.perf_counter() - start,
        "tensors": tensor_reports,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"original_size={report['original_size']}")
    print(f"resized_size={report['resized_size']}")
    print(f"grid_thw={report['grid_thw']}")
    print(f"sequence_length={report['sequence_length']}")
    print(f"image_token_id={report['image_token_id']}")
    print(f"image_token_spans={report['image_token_spans']}")
    print(f"mm_type_one_spans={report['mm_type_one_spans']}")
    print(f"prefix_input_ids={report['prefix_input_ids']}")
    print(f"suffix_input_ids={report['suffix_input_ids']}")
    print(f"pixel_error={manual_pixel_error:.3e}")
    print(f"fused_error={fused_error:.3e}")


if __name__ == "__main__":
    main()
