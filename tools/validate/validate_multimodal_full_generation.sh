#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/work/hunyuanocr/HunyuanOCR-ncnn"
REFERENCE_PYTHON="$HOME/work/hunyuanocr/.venv-reference/bin/python"
NCNN_PREFIX="$HOME/.local/ncnn-cpu-ropefix-rmsnorm"
RECOVERY_DIR="$HOME/hunyuanocr-recovery"
CPP_SOURCE="$ROOT/tests/multimodal_full_generation/cpp"
CPP_BUILD="$ROOT/tests/multimodal_full_generation/build-ropefix-rmsnorm-clean"
CPP_BINARY="$CPP_BUILD/multimodal_full_generation"

mkdir -p "$RECOVERY_DIR"
cd "$ROOT"

for file in \
  "$REFERENCE_PYTHON" \
  "$CPP_SOURCE/main.cpp" \
  "$CPP_SOURCE/multimodal_input.cpp" \
  "$CPP_SOURCE/CMakeLists.txt" \
  "$ROOT/third_party/stb/stb_image.h" \
  "$ROOT/docs/multimodal_prefill_input_reference.json"
do
  if [ ! -s "$file" ]; then
    echo "Missing required file: $file"
    exit 1
  fi
done

echo "============================================================"
echo "1. Audit the captured multimodal processor contract"
echo "============================================================"

"$REFERENCE_PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path


root = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
report = json.loads(
    (root / "docs/multimodal_prefill_input_reference.json").read_text(
        encoding="utf-8"
    )
)
assert report["image_path"] == "tests/assets/ocr_smoke_en.png"
assert report["original_size"] == [320, 768]
assert report["resized_size"] == [352, 800]
assert report["grid_thw"] == [[1, 22, 50]]
assert report["sequence_length"] == 313
assert report["image_token_id"] == 120120
assert report["image_token_spans"] == [[2, 290]]
assert report["image_token_count"] == 288
assert report["fused_prefill_hidden_max_abs_error"] == 0.0

image = root / report["image_path"]
assert hashlib.sha256(image.read_bytes()).hexdigest() == report["image_sha256"]

raw = root / "artifacts/multimodal_prefill_input/reference"
expected_sizes = {
    "original_rgb_u8.bin": 320 * 768 * 3,
    "resized_rgb_u8.bin": 352 * 800 * 3,
    "pixel_values_f32.bin": 1100 * 768 * 4,
    "image_grid_thw_i64.bin": 3 * 8,
    "input_ids_i64.bin": 313 * 8,
    "text_embeddings_f32.bin": 313 * 1024 * 4,
    "image_features_f32.bin": 288 * 1024 * 4,
    "fused_embeddings_f32.bin": 313 * 1024 * 4,
}
for name, expected_size in expected_sizes.items():
    assert (raw / name).stat().st_size == expected_size, name

for name in ["vision_patch_embed", "vision_patch_merger"]:
    assert (root / "artifacts" / name / f"{name}.ncnn.param").is_file()
for layer in range(27):
    name = f"vision_block{layer}"
    assert (root / "artifacts" / name / f"{name}.ncnn.param").is_file()
for layer in range(24):
    name = f"decoder_layer{layer}_prefill_kv"
    assert (root / "artifacts" / name / f"{name}.ncnn.param").is_file()

print("Processor contract: PNG 768x320 -> RGB 800x352 -> grid [1,22,50]")
print("Prompt contract: 313 tokens, image span [2,290), 288 embeddings")
print("Model inventory: 29 vision components and 24 prefill layers")
PY

echo
echo "============================================================"
echo "2. Configure and build from the persistent clean directory"
echo "============================================================"

cmake \
  -S "$CPP_SOURCE" \
  -B "$CPP_BUILD" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$NCNN_PREFIX/lib/cmake/ncnn" \
  2>&1 | tee "$RECOVERY_DIR/multimodal_full_generation_cmake.log"

cmake \
  --build "$CPP_BUILD" \
  --parallel 8 \
  2>&1 | tee "$RECOVERY_DIR/multimodal_full_generation_build.log"

echo
echo "============================================================"
echo "3. Run packed and unpacked image-to-text validation"
echo "============================================================"

/usr/bin/time -v \
  "$CPP_BINARY" "$ROOT" 1 \
  2>&1 | tee "$RECOVERY_DIR/multimodal_full_generation_packed.log"

/usr/bin/time -v \
  "$CPP_BINARY" "$ROOT" 0 \
  2>&1 | tee "$RECOVERY_DIR/multimodal_full_generation_unpacked.log"

echo
echo "============================================================"
echo "4. Create the machine-readable validation report"
echo "============================================================"

"$REFERENCE_PYTHON" - <<'PY'
import json
import re
from pathlib import Path


root = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
recovery = Path.home() / "hunyuanocr-recovery"
expected_tokens = [
    93892, 5112, 206, 1717, 21, 185,
    18009, 15613, 16678, 21836, 120007,
]
expected_text = "HELLO 2026\nNCNN CPU TEST"


def number(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(pattern)
    return float(match.group(1))


def elapsed_seconds(text: str) -> float:
    match = re.search(
        r"Elapsed \(wall clock\) time .*: ([0-9:.]+)", text
    )
    if match is None:
        raise AssertionError("elapsed wall-clock time")
    parts = [float(part) for part in match.group(1).split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise AssertionError(parts)


def parse(mode: str) -> dict[str, object]:
    path = recovery / f"multimodal_full_generation_{mode}.log"
    text = path.read_text(encoding="utf-8")
    assert "Full ncnn prefill + autoregressive generation passed." in text
    assert "EOS reached      : true" in text
    assert f"Generated text:\n{expected_text}\n" in text

    token_match = re.search(r"Generated tokens: ([0-9 ]+)", text)
    assert token_match is not None
    tokens = [int(value) for value in token_match.group(1).split()]
    assert tokens == expected_tokens

    steps = re.findall(
        r"Step\s+([0-9]+):.*?logits_max=([0-9.eE+-]+), prefix=([0-9.eE+-]+)",
        text,
    )
    assert len(steps) == 10
    assert all(float(prefix) == 0.0 for _, _, prefix in steps)

    result = {
        "tokens": tokens,
        "text": expected_text,
        "original_rgb_max_abs_error": number(
            r"original RGB: max=([0-9.eE+-]+)", text
        ),
        "resized_rgb_max_abs_error": number(
            r"resized RGB: max=([0-9.eE+-]+)", text
        ),
        "pixel_values_max_abs_error": number(
            r"pixel values: max=([0-9.eE+-]+)", text
        ),
        "vision_embedding_max_abs_error": number(
            r"vision embeddings: max=([0-9.eE+-]+)", text
        ),
        "fused_hidden_max_abs_error": number(
            r"fused hidden: max=([0-9.eE+-]+)", text
        ),
        "prefill_hidden_max_abs_error": number(
            r"Maximum hidden error\s+: ([0-9.eE+-]+)", text
        ),
        "prefill_cache_max_abs_error": number(
            r"Maximum cache error\s+: ([0-9.eE+-]+)", text
        ),
        "prefill_final_hidden_max_abs_error": number(
            r"Final hidden error\s+: ([0-9.eE+-]+)", text
        ),
        "prefill_final_norm_max_abs_error": number(
            r"Final norm error\s+: ([0-9.eE+-]+)", text
        ),
        "prefill_logits_max_abs_error": number(
            r"Prefill logits error\s+: ([0-9.eE+-]+)", text
        ),
        "decode_logits_max_abs_error": max(
            float(logits) for _, logits, _ in steps
        ),
        "kv_prefix_max_abs_error": max(
            float(prefix) for _, _, prefix in steps
        ),
        "elapsed_wall_seconds": elapsed_seconds(text),
        "maximum_resident_kib": int(number(
            r"Maximum resident set size \(kbytes\): ([0-9]+)", text
        )),
    }
    assert result["original_rgb_max_abs_error"] == 0.0
    assert result["resized_rgb_max_abs_error"] == 0.0
    assert result["pixel_values_max_abs_error"] <= 1.0e-6
    assert result["vision_embedding_max_abs_error"] <= 1.0e-1
    assert result["fused_hidden_max_abs_error"] <= 1.0e-1
    assert result["prefill_hidden_max_abs_error"] <= 5.0e-1
    assert result["prefill_cache_max_abs_error"] <= 5.0e-1
    assert result["prefill_logits_max_abs_error"] <= 5.0e-2
    assert result["decode_logits_max_abs_error"] <= 5.0e-2
    assert result["kv_prefix_max_abs_error"] == 0.0
    return result


reference = json.loads(
    (root / "docs/multimodal_prefill_input_reference.json").read_text(
        encoding="utf-8"
    )
)
report = {
    "status": "passed",
    "image_sha256": reference["image_sha256"],
    "image_grid_thw": [1, 22, 50],
    "prefill_hidden_shape": [1, 313, 1024],
    "image_token_span": [2, 290],
    "image_embedding_shape": [1, 288, 1024],
    "packed": parse("packed"),
    "unpacked": parse("unpacked"),
}
path = root / "docs/multimodal_full_generation_validation.json"
path.write_text(
    json.dumps(report, ensure_ascii=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=True, indent=2))
PY

echo
echo "Packed and unpacked multimodal image-to-text validation passed."
