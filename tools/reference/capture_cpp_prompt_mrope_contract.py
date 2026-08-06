from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLRotaryEmbedding,
)


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
IMAGE_PATH = PROJECT_DIR / "tests/assets/ocr_smoke_en.png"
MULTIMODAL_REFERENCE = (
    PROJECT_DIR / "artifacts/multimodal_prefill_input/reference"
)
PREFILL_REFERENCE = PROJECT_DIR / "artifacts/decoder_layer0_prefill_kv/reference"
REPORT_PATH = PROJECT_DIR / "docs/cpp_prompt_mrope_reference.json"

SEQUENCE_LENGTH = 313
MROPE_AXES = 4
HEAD_DIM = 128
IMAGE_TOKEN_ID = 120120
IMAGE_START = 2
IMAGE_END = 290
MERGE_SIZE = 2
SPATIAL_PATCH_SIZE = 1


def load_smoke_module():
    path = PROJECT_DIR / "tools/reference/run_reference_smoke.py"
    spec = importlib.util.spec_from_file_location("reference_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_binary(path: Path, dtype, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if value.size != expected:
        raise RuntimeError(f"Unexpected element count for {path}: {value.size}")
    return value.reshape(shape)


def maximum_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float64) - expected)))


def build_position_ids(
    sequence_length: int,
    image_grid_thw: list[int],
) -> np.ndarray:
    position_ids = np.broadcast_to(
        np.arange(sequence_length, dtype=np.int64),
        (MROPE_AXES, sequence_length),
    ).copy()
    _, grid_h, grid_w = image_grid_thw
    llm_h = grid_h // MERGE_SIZE // SPATIAL_PATCH_SIZE
    llm_w = grid_w // MERGE_SIZE // SPATIAL_PATCH_SIZE
    grid_start = IMAGE_START + 1
    index = grid_start
    for height in range(llm_h):
        for width in range(llm_w + 1):
            position_ids[1, index] = width
            position_ids[2, index] = height
            position_ids[3, index] = 0
            index += 1
    if index != IMAGE_END - 1:
        raise RuntimeError(f"Unexpected image position contract: {index=}")
    return position_ids[:, None, :]


def main() -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(9)
    smoke = load_smoke_module()
    config = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR), backend="pil", local_files_only=True
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(IMAGE_PATH)},
                {"type": "text", "text": smoke.PROMPT},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].cpu().numpy().astype(np.int64)
    attention_mask = inputs["attention_mask"].cpu().numpy().astype(np.int64)
    mm_types = inputs["mm_token_type_ids"].cpu().numpy().astype(np.int64)
    image_grid = inputs["image_grid_thw"][0].cpu().tolist()
    if input_ids.shape != (1, SEQUENCE_LENGTH):
        raise RuntimeError(f"Unexpected input shape: {input_ids.shape}")

    position_ids = build_position_ids(SEQUENCE_LENGTH, image_grid)
    causal_mask = np.full(
        (SEQUENCE_LENGTH, SEQUENCE_LENGTH),
        np.finfo(np.float32).min,
        dtype=np.float32,
    )
    causal_mask[np.tril_indices(SEQUENCE_LENGTH)] = 0.0

    rotary = HunYuanVLRotaryEmbedding(config.text_config)
    dummy = torch.zeros((1, SEQUENCE_LENGTH, 1), dtype=torch.float32)
    torch_positions = torch.from_numpy(position_ids)
    rope_cos, rope_sin = rotary(dummy, torch_positions)
    rope_cos_np = rope_cos.cpu().numpy().astype(np.float32)
    rope_sin_np = rope_sin.cpu().numpy().astype(np.float32)

    expected_ids = load_binary(
        MULTIMODAL_REFERENCE / "input_ids_i64.bin",
        np.int64,
        (1, SEQUENCE_LENGTH),
    )
    expected_attention = load_binary(
        MULTIMODAL_REFERENCE / "attention_mask_i64.bin",
        np.int64,
        (1, SEQUENCE_LENGTH),
    )
    expected_mm = load_binary(
        MULTIMODAL_REFERENCE / "mm_token_type_ids_i64.bin",
        np.int64,
        (1, SEQUENCE_LENGTH),
    )
    expected_positions = load_binary(
        MULTIMODAL_REFERENCE / "position_ids_i64.bin",
        np.int64,
        (MROPE_AXES, 1, SEQUENCE_LENGTH),
    )
    expected_causal = load_binary(
        PREFILL_REFERENCE / "layer0_attention_mask_f32.bin",
        np.float32,
        (SEQUENCE_LENGTH, SEQUENCE_LENGTH),
    )
    expected_cos = load_binary(
        PREFILL_REFERENCE / "layer0_position_embeddings_0_f32.bin",
        np.float32,
        (MROPE_AXES, 1, SEQUENCE_LENGTH, HEAD_DIM),
    )
    expected_sin = load_binary(
        PREFILL_REFERENCE / "layer0_position_embeddings_1_f32.bin",
        np.float32,
        (MROPE_AXES, 1, SEQUENCE_LENGTH, HEAD_DIM),
    )

    exact = {
        "input_ids": bool(np.array_equal(input_ids, expected_ids)),
        "attention_mask": bool(np.array_equal(attention_mask, expected_attention)),
        "mm_token_type_ids": bool(np.array_equal(mm_types, expected_mm)),
        "position_ids": bool(np.array_equal(position_ids, expected_positions)),
        "causal_mask": bool(np.array_equal(causal_mask, expected_causal)),
        "rope_cos": bool(np.array_equal(rope_cos_np, expected_cos)),
        "rope_sin": bool(np.array_equal(rope_sin_np, expected_sin)),
    }
    if not all(exact.values()):
        raise RuntimeError(f"Phase 3C reference mismatch: {exact}")

    image_positions = np.flatnonzero(mm_types[0] == 1).tolist()
    tokenizer_json = json.loads(
        (MODEL_DIR / "tokenizer.json").read_text(encoding="utf-8")
    )
    report = {
        "phase": "3C",
        "prompt": smoke.PROMPT,
        "prompt_utf8_hex": smoke.PROMPT.encode("utf-8").hex(),
        "rendered_chat_template": rendered,
        "sequence_length": SEQUENCE_LENGTH,
        "input_ids": input_ids[0].tolist(),
        "prefix_input_ids": input_ids[0, :IMAGE_START].tolist(),
        "image_token_id": IMAGE_TOKEN_ID,
        "image_token_span": [image_positions[0], image_positions[-1] + 1],
        "image_token_count": len(image_positions),
        "suffix_input_ids": input_ids[0, IMAGE_END:].tolist(),
        "attention_mask_unique": np.unique(attention_mask).tolist(),
        "mm_token_type_ids_unique": np.unique(mm_types).tolist(),
        "position_axis_ranges": [
            [int(axis.min()), int(axis.max())] for axis in position_ids[:, 0]
        ],
        "image_grid_thw": image_grid,
        "rope_parameters": config.text_config.rope_parameters,
        "rope_inv_freq_f32": rotary.inv_freq.cpu().tolist(),
        "tokenizer": {
            "model_type": tokenizer_json["model"]["type"],
            "vocabulary_size": len(tokenizer_json["model"]["vocab"]),
            "merge_count": len(tokenizer_json["model"]["merges"]),
            "normalizer": tokenizer_json.get("normalizer"),
            "pre_tokenizer": tokenizer_json.get("pre_tokenizer"),
            "decoder": tokenizer_json.get("decoder"),
        },
        "sha256": {
            "input_ids": sha256_bytes(input_ids.tobytes()),
            "attention_mask": sha256_bytes(attention_mask.tobytes()),
            "mm_token_type_ids": sha256_bytes(mm_types.tobytes()),
            "position_ids": sha256_bytes(position_ids.tobytes()),
            "causal_mask": sha256_bytes(causal_mask.tobytes()),
            "rope_cos": sha256_bytes(rope_cos_np.tobytes()),
            "rope_sin": sha256_bytes(rope_sin_np.tobytes()),
        },
        "captured_reference_exact_match": exact,
        "maximum_abs_difference": {
            "rope_cos": maximum_difference(rope_cos_np, expected_cos),
            "rope_sin": maximum_difference(rope_sin_np, expected_sin),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"sequence_length={SEQUENCE_LENGTH}")
    print(f"image_token_span=[{image_positions[0]}, {image_positions[-1] + 1})")
    print(f"suffix_input_ids={report['suffix_input_ids']}")
    print(f"all_reference_fields_exact={all(exact.values())}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
