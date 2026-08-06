#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/work/hunyuanocr/HunyuanOCR-ncnn"
REFERENCE_PYTHON="$HOME/work/hunyuanocr/.venv-reference/bin/python"
NCNN_PREFIX="$HOME/.local/ncnn-cpu-ropefix-rmsnorm"
RECOVERY_DIR="$HOME/hunyuanocr-recovery/phase4a"
BUILD_DIR="$ROOT/build-phase4a-clean"
INSTALL_DIR="$RECOVERY_DIR/install"
MODEL_DIR="$ROOT/artifacts"
IMAGE="$ROOT/tests/assets/ocr_smoke_en.png"
CLI="$BUILD_DIR/hunyuanocr_cli"
PARITY="$BUILD_DIR/tests/multimodal_full_generation/multimodal_full_generation"
CONTRACT="$BUILD_DIR/tests/multimodal_full_generation/prompt_inputs_contract"

mkdir -p "$RECOVERY_DIR" "$INSTALL_DIR"
cd "$ROOT"

echo "============================================================"
echo "1. Export and audit the runtime model manifest"
echo "============================================================"

"$REFERENCE_PYTHON" tools/export/export_tokenizer_vocab.py \
  2>&1 | tee "$RECOVERY_DIR/tokenizer_export.log"
"$REFERENCE_PYTHON" tools/export/export_runtime_manifest.py \
  2>&1 | tee "$RECOVERY_DIR/manifest_export.log"

if grep -R -nE \
  'reference/|/reference|input_ids_i64|attention_mask_i64|position_ids_i64|_f32\.bin' \
  include src app
then
  echo "Product runtime source still references captured tensors"
  exit 1
fi
echo "Product source captured-reference audit: clean"

echo
echo "============================================================"
echo "2. Configure, build, and install the root CMake project"
echo "============================================================"

cmake \
  -S "$ROOT" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -Dncnn_DIR="$NCNN_PREFIX/lib/cmake/ncnn" \
  -DHUNYUANOCR_BUILD_PARITY_TESTS=ON \
  2>&1 | tee "$RECOVERY_DIR/cmake_configure.log"

cmake --build "$BUILD_DIR" --parallel 8 \
  2>&1 | tee "$RECOVERY_DIR/cmake_build.log"
cmake --install "$BUILD_DIR" \
  2>&1 | tee "$RECOVERY_DIR/cmake_install.log"

test -s "$INSTALL_DIR/include/hunyuanocr/runtime.h"
test -s "$INSTALL_DIR/lib/libhunyuanocr_runtime.a"
test -x "$INSTALL_DIR/bin/hunyuanocr_cli"
test -s "$INSTALL_DIR/lib/cmake/HunyuanOCR/HunyuanOCRConfig.cmake"

cmake \
  -S "$ROOT/tests/runtime_api_consumer" \
  -B "$RECOVERY_DIR/runtime-api-consumer-build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHunyuanOCR_DIR="$INSTALL_DIR/lib/cmake/HunyuanOCR" \
  -Dncnn_DIR="$NCNN_PREFIX/lib/cmake/ncnn" \
  2>&1 | tee "$RECOVERY_DIR/consumer_cmake_configure.log"
cmake \
  --build "$RECOVERY_DIR/runtime-api-consumer-build" \
  --parallel 4 \
  2>&1 | tee "$RECOVERY_DIR/consumer_cmake_build.log"
"$RECOVERY_DIR/runtime-api-consumer-build/runtime_api_consumer"

echo
echo "============================================================"
echo "3. Run generated-input and full SHA-256 contract checks"
echo "============================================================"

"$CONTRACT" "$ROOT" \
  2>&1 | tee "$RECOVERY_DIR/prompt_inputs_contract.log"

/usr/bin/time -v \
  "$CLI" \
  --model-dir "$MODEL_DIR" \
  --image "$IMAGE" \
  --packing 0 \
  --threads 9 \
  --max-new-tokens 1 \
  --verify sha256 \
  2>&1 | tee "$RECOVERY_DIR/cli_sha256_verification.log"

echo
echo "============================================================"
echo "4. Run reference-free packed and unpacked OCR CLI"
echo "============================================================"

/usr/bin/time -v \
  "$CLI" \
  --model-dir "$MODEL_DIR" \
  --image "$IMAGE" \
  --packing 1 \
  --threads 9 \
  --max-new-tokens 32 \
  --verify size \
  2>&1 | tee "$RECOVERY_DIR/cli_packed.log"

/usr/bin/time -v \
  "$CLI" \
  --model-dir "$MODEL_DIR" \
  --image "$IMAGE" \
  --packing 0 \
  --threads 9 \
  --max-new-tokens 32 \
  --verify size \
  2>&1 | tee "$RECOVERY_DIR/cli_unpacked.log"

echo
echo "============================================================"
echo "5. Run the independent reference-backed parity mode"
echo "============================================================"

"$PARITY" "$ROOT" 1 \
  2>&1 | tee "$RECOVERY_DIR/parity_packed.log"
"$PARITY" "$ROOT" 0 \
  2>&1 | tee "$RECOVERY_DIR/parity_unpacked.log"

echo
echo "============================================================"
echo "6. Write the machine-readable Phase 4A report"
echo "============================================================"

"$REFERENCE_PYTHON" - <<'PY'
import json
import re
from pathlib import Path


root = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
recovery = Path.home() / "hunyuanocr-recovery/phase4a"
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


def parse_cli(mode: str) -> dict[str, object]:
    text = (recovery / f"cli_{mode}.log").read_text(encoding="utf-8")
    assert "EOS reached     : true" in text
    assert f"Generated text:\n{expected_text}\n" in text
    match = re.search(r"Generated tokens: ([0-9 ]+)", text)
    assert match is not None
    tokens = [int(value) for value in match.group(1).split()]
    assert tokens == expected_tokens
    return {
        "tokens": tokens,
        "text": expected_text,
        "load_seconds": number(r"Load seconds\s+: ([0-9.]+)", text),
        "input_seconds": number(r"Input seconds\s+: ([0-9.]+)", text),
        "prefill_seconds": number(r"Prefill seconds\s+: ([0-9.]+)", text),
        "decode_seconds": number(r"Decode seconds\s+: ([0-9.]+)", text),
        "runtime_seconds": number(r"Runtime seconds\s+: ([0-9.]+)", text),
        "peak_rss_kib": int(number(r"Peak RSS KiB\s+: ([0-9]+)", text)),
    }


def parse_parity(mode: str) -> dict[str, object]:
    text = (recovery / f"parity_{mode}.log").read_text(encoding="utf-8")
    assert "Full ncnn prefill + autoregressive generation passed." in text
    assert f"Generated text:\n{expected_text}\n" in text
    match = re.search(r"Generated tokens: ([0-9 ]+)", text)
    assert match is not None
    tokens = [int(value) for value in match.group(1).split()]
    assert tokens == expected_tokens
    return {"status": "passed", "tokens": tokens, "text": expected_text}


manifest = json.loads(
    (root / "docs/runtime_manifest.json").read_text(encoding="utf-8")
)
assert manifest["file_count"] == 170
groups = {
    "text_embedding": 0,
    "lm_head": 0,
    "vision": 0,
    "prefill": 0,
    "dynamic_decode": 0,
    "tokenizer": 0,
}
for entry in manifest["entries"]:
    path = entry["path"]
    size = entry["bytes"]
    if path.startswith("text_embedding/"):
        groups["text_embedding"] += size
    elif path.startswith("lm_head/"):
        groups["lm_head"] += size
    elif path.startswith("vision_"):
        groups["vision"] += size
    elif "_prefill_kv/" in path:
        groups["prefill"] += size
    elif "_decode_dynamic/" in path:
        groups["dynamic_decode"] += size
    elif path.startswith("tokenizer/"):
        groups["tokenizer"] += size

sha_text = (recovery / "cli_sha256_verification.log").read_text(
    encoding="utf-8"
)
assert "Generated tokens: 93892" in sha_text
assert "Exit status: 0" in sha_text
contract = (recovery / "prompt_inputs_contract.log").read_text(
    encoding="utf-8"
)
assert "prompt_mrope_contract_passed=true" in contract

report = {
    "phase": "4A",
    "status": "passed",
    "product_runtime_captured_reference_inputs": False,
    "root_cmake_build": "passed",
    "install_tree": "passed",
    "installed_package_consumer": "passed",
    "manifest": {
        "format": manifest["format"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "total_gib": manifest["total_gib"],
        "cpp_sha256_verification": "passed",
        "component_bytes": groups,
    },
    "packed_cli": parse_cli("packed"),
    "unpacked_cli": parse_cli("unpacked"),
    "packed_parity": parse_parity("packed"),
    "unpacked_parity": parse_parity("unpacked"),
}
path = root / "docs/product_runtime_validation.json"
path.write_text(
    json.dumps(report, ensure_ascii=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=True, indent=2))
PY

echo
echo "Phase 4A product runtime validation passed."
