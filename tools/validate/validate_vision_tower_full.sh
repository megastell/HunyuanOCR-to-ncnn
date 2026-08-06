#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/work/hunyuanocr/HunyuanOCR-ncnn"
REFERENCE_PYTHON="$HOME/work/hunyuanocr/.venv-reference/bin/python"
NCNN_PREFIX="$HOME/.local/ncnn-cpu-ropefix-rmsnorm"
RECOVERY_DIR="$HOME/hunyuanocr-recovery"

CPP_SOURCE="$ROOT/tests/vision_tower_full/cpp"
CPP_BUILD="$ROOT/tests/vision_tower_full/build-ropefix-rmsnorm"
CPP_BINARY="$CPP_BUILD/vision_tower_full"

mkdir -p "$RECOVERY_DIR"
cd "$ROOT"

for file in \
  "$REFERENCE_PYTHON" \
  "$CPP_SOURCE/main.cpp" \
  "$CPP_SOURCE/CMakeLists.txt" \
  "$ROOT/docs/vision_tower_full_reference.json"
do
  if [ ! -s "$file" ]; then
    echo "Missing required file: $file"
    exit 1
  fi
done

echo "============================================================"
echo "1. Audit reference, reports, and ncnn attention layout"
echo "============================================================"

"$REFERENCE_PYTHON" - <<'PY'
import json
from pathlib import Path


root = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
reference = json.loads(
    (root / "docs/vision_tower_full_reference.json").read_text(
        encoding="utf-8"
    )
)
assert reference["layer_count"] == 27
assert reference["patch_embedding"]["expected_output"]["shape"] == [
    1, 1100, 1152
]
assert reference["patch_merger"]["tensors"]["expected_output"]["shape"] == [
    1, 288, 1024
]
assert reference["rotary_module_count"] == 0

components = ["vision_patch_embed"]
components.extend(f"vision_block{layer}" for layer in range(27))
components.append("vision_patch_merger")

for name in components:
    report_path = root / "docs" / f"{name}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["component"] == name
    assert report["eager_metrics"]["mean_abs_error"] <= 2.0e-6
    assert report["torchscript_metrics"]["mean_abs_error"] <= 2.0e-6

for layer in range(27):
    name = f"vision_block{layer}"
    param_path = root / "artifacts" / name / f"{name}.ncnn.param"
    text = param_path.read_text(encoding="utf-8")
    assert text.count("vision_head_layout_") == 3
    assert text.count("SDPA ") == 1
    for line in text.splitlines():
        if line.startswith("Reshape ") and (
            " 0=72 1=16 2=1100" in line
            or " 0=1152 1=1100" in line
        ):
            assert " 12=" not in line
            assert " 13=" not in line

print("Reference contract: 27 blocks, no vision RoPE, output [1,288,1024]")
print("Conversion reports: 29/29")
print("Corrected ncnn SDPA layouts: 27/27")
PY

echo
echo "============================================================"
echo "2. Configure and build the C++ vision chain"
echo "============================================================"

cmake \
  -S "$CPP_SOURCE" \
  -B "$CPP_BUILD" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$NCNN_PREFIX/lib/cmake/ncnn" \
  2>&1 | tee "$RECOVERY_DIR/vision_tower_full_cmake.log"

cmake \
  --build "$CPP_BUILD" \
  --parallel 8 \
  2>&1 | tee "$RECOVERY_DIR/vision_tower_full_build.log"

echo
echo "============================================================"
echo "3. Run packed and unpacked validation"
echo "============================================================"

/usr/bin/time -v \
  "$CPP_BINARY" "$ROOT" 1 \
  2>&1 | tee "$RECOVERY_DIR/vision_tower_full_packed.log"

/usr/bin/time -v \
  "$CPP_BINARY" "$ROOT" 0 \
  2>&1 | tee "$RECOVERY_DIR/vision_tower_full_unpacked.log"

echo
echo "============================================================"
echo "4. Compare final vision embeddings"
echo "============================================================"

"$REFERENCE_PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
output = root / "artifacts/vision_tower_full/output"
reference_path = (
    root
    / "artifacts/vision_patch_merger/reference/expected_output_f32.bin"
)

expected = np.fromfile(reference_path, dtype=np.float32)
packed = np.fromfile(
    output / "vision_embeddings_packed_f32.bin", dtype=np.float32
)
unpacked = np.fromfile(
    output / "vision_embeddings_unpacked_f32.bin", dtype=np.float32
)
assert expected.size == packed.size == unpacked.size == 288 * 1024


def metrics(actual: np.ndarray, target: np.ndarray) -> dict[str, float]:
    actual64 = actual.astype(np.float64)
    target64 = target.astype(np.float64)
    difference = np.abs(actual64 - target64)
    denominator = np.linalg.norm(actual64) * np.linalg.norm(target64)
    cosine = float(np.dot(actual64, target64) / denominator)
    return {
        "maximum_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "cosine_similarity": cosine,
    }


packed_metrics = metrics(packed, expected)
unpacked_metrics = metrics(unpacked, expected)
cross_metrics = metrics(packed, unpacked)

for result in (packed_metrics, unpacked_metrics):
    assert result["maximum_abs_error"] <= 1.0e-1
    assert result["mean_abs_error"] <= 1.0e-3
    assert result["cosine_similarity"] >= 0.99999

report = {
    "status": "passed",
    "shape": [1, 288, 1024],
    "packed": packed_metrics,
    "unpacked": unpacked_metrics,
    "packed_vs_unpacked": cross_metrics,
    "packed_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
    "unpacked_sha256": hashlib.sha256(unpacked.tobytes()).hexdigest(),
    "interface": {
        "input_pixel_values": [1100, 768],
        "input_image_grid_thw": [1, 3],
        "output_vision_embeddings": [1, 288, 1024],
        "next_consumer": "multimodal embedding placement",
    },
}

report_path = root / "docs/vision_tower_full_validation.json"
report_path.write_text(
    json.dumps(report, ensure_ascii=True, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(report, ensure_ascii=True, indent=2))
PY

echo
echo "Full packed and unpacked ncnn vision validation passed."
