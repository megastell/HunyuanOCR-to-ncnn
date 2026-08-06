#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/work/hunyuanocr/HunyuanOCR-ncnn"
REFERENCE_PYTHON="$HOME/work/hunyuanocr/.venv-reference/bin/python"
NCNN_PREFIX="$HOME/.local/ncnn-cpu-ropefix-rmsnorm"
RECOVERY_DIR="$HOME/hunyuanocr-recovery/phase3c"
CPP_SOURCE="$ROOT/tests/multimodal_full_generation/cpp"
CPP_BUILD="$ROOT/tests/multimodal_full_generation/build-phase3c-clean"
CPP_BINARY="$CPP_BUILD/multimodal_full_generation"
CONTRACT_BINARY="$CPP_BUILD/prompt_inputs_contract"

mkdir -p "$RECOVERY_DIR"
cd "$ROOT"

for file in \
  "$REFERENCE_PYTHON" \
  "$ROOT/tools/export/export_tokenizer_vocab.py" \
  "$ROOT/tools/reference/capture_cpp_prompt_mrope_contract.py" \
  "$CPP_SOURCE/prompt_inputs.cpp" \
  "$CPP_SOURCE/prompt_inputs.h" \
  "$CPP_SOURCE/prompt_inputs_contract.cpp"
do
  if [ ! -s "$file" ]; then
    echo "Missing required file: $file"
    exit 1
  fi
done

echo "============================================================"
echo "1. Export tokenizer assets and recapture the Phase 3C contract"
echo "============================================================"

"$REFERENCE_PYTHON" \
  tools/export/export_tokenizer_vocab.py \
  2>&1 | tee "$RECOVERY_DIR/tokenizer_export.log"

"$REFERENCE_PYTHON" \
  tools/reference/capture_cpp_prompt_mrope_contract.py \
  2>&1 | tee "$RECOVERY_DIR/reference_capture.log"

echo
echo "============================================================"
echo "2. Audit the runtime source and build from a persistent directory"
echo "============================================================"

if grep -E \
  'input_ids_i64|attention_mask_i64|mm_token_type_ids_i64|position_ids_i64|layer0_attention_mask_f32|layer0_position_embeddings_[01]_f32' \
  "$CPP_SOURCE/main.cpp" "$CPP_SOURCE/multimodal_input.cpp"
then
  echo "Captured prompt or positional input remains in the runtime path"
  exit 1
fi

cmake \
  -S "$CPP_SOURCE" \
  -B "$CPP_BUILD" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$NCNN_PREFIX/lib/cmake/ncnn" \
  2>&1 | tee "$RECOVERY_DIR/cmake_configure.log"

cmake \
  --build "$CPP_BUILD" \
  --parallel 8 \
  2>&1 | tee "$RECOVERY_DIR/cmake_build.log"

echo
echo "============================================================"
echo "3. Compare generated C++ prompt and position inputs"
echo "============================================================"

"$CONTRACT_BINARY" "$ROOT" \
  2>&1 | tee "$RECOVERY_DIR/prompt_inputs_contract.log"

echo
echo "============================================================"
echo "4. Run packed and unpacked image-to-text generation"
echo "============================================================"

/usr/bin/time -v \
  "$CPP_BINARY" "$ROOT" 1 \
  2>&1 | tee "$RECOVERY_DIR/ncnn_packed.log"

/usr/bin/time -v \
  "$CPP_BINARY" "$ROOT" 0 \
  2>&1 | tee "$RECOVERY_DIR/ncnn_unpacked.log"

echo
echo "============================================================"
echo "5. Write the machine-readable Phase 3C report"
echo "============================================================"

"$REFERENCE_PYTHON" - <<'PY'
import json
import re
from pathlib import Path


root = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
recovery = Path.home() / "hunyuanocr-recovery/phase3c"
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


def parse_run(mode: str) -> dict[str, object]:
    text = (recovery / f"ncnn_{mode}.log").read_text(encoding="utf-8")
    assert "prompt input ids: generated in C++ (313)" in text
    assert "Full ncnn prefill + autoregressive generation passed." in text
    assert "EOS reached      : true" in text
    assert f"Generated text:\n{expected_text}\n" in text
    token_match = re.search(r"Generated tokens: ([0-9 ]+)", text)
    assert token_match is not None
    tokens = [int(value) for value in token_match.group(1).split()]
    assert tokens == expected_tokens
    steps = re.findall(
        r"Step\s+([0-9]+):.*?logits_max=([0-9.eE+-]+), "
        r"prefix=([0-9.eE+-]+)",
        text,
    )
    assert len(steps) == 10
    assert all(float(prefix) == 0.0 for _, _, prefix in steps)
    return {
        "tokens": tokens,
        "text": expected_text,
        "prefill_token": int(number(r"Prefill token\s+: ([0-9]+)", text)),
        "vision_embedding_max_abs_error": number(
            r"vision embeddings: max=([0-9.eE+-]+)", text
        ),
        "prefill_hidden_max_abs_error": number(
            r"Maximum hidden error\s+: ([0-9.eE+-]+)", text
        ),
        "prefill_cache_max_abs_error": number(
            r"Maximum cache error\s+: ([0-9.eE+-]+)", text
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
        "maximum_resident_kib": int(number(
            r"Maximum resident set size \(kbytes\): ([0-9]+)", text
        )),
    }


contract_text = (recovery / "prompt_inputs_contract.log").read_text(
    encoding="utf-8"
)
for field in [
    "input_ids_exact",
    "attention_mask_exact",
    "mm_token_type_ids_exact",
    "position_ids_exact",
    "causal_mask_exact",
    "decode_masks_exact",
    "prompt_mrope_contract_passed",
]:
    assert f"{field}=true" in contract_text

reference = json.loads(
    (root / "docs/cpp_prompt_mrope_reference.json").read_text(
        encoding="utf-8"
    )
)
assert all(reference["captured_reference_exact_match"].values())
report = {
    "phase": "3C",
    "status": "passed",
    "runtime_captured_prompt_or_position_inputs": False,
    "prompt": reference["prompt"],
    "sequence_length": 313,
    "image_grid_thw": [1, 22, 50],
    "image_token_span": [2, 290],
    "input_ids_exact": True,
    "attention_mask_exact": True,
    "mm_token_type_ids_exact": True,
    "position_ids_exact": True,
    "causal_mask_exact": True,
    "prefill_rope_cos_max_abs_error": number(
        r"rope_cos_max_abs=([0-9.eE+-]+)", contract_text
    ),
    "prefill_rope_sin_max_abs_error": number(
        r"rope_sin_max_abs=([0-9.eE+-]+)", contract_text
    ),
    "decode_rope_cos_max_abs_error": number(
        r"decode_rope_cos_max_abs=([0-9.eE+-]+)", contract_text
    ),
    "decode_rope_sin_max_abs_error": number(
        r"decode_rope_sin_max_abs=([0-9.eE+-]+)", contract_text
    ),
    "packed": parse_run("packed"),
    "unpacked": parse_run("unpacked"),
}
assert report["packed"]["prefill_token"] == 93892
assert report["unpacked"]["prefill_token"] == 93892
path = root / "docs/cpp_prompt_mrope_validation.json"
path.write_text(
    json.dumps(report, ensure_ascii=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=True, indent=2))
PY

echo
echo "Phase 3C C++ prompt, mRoPE, and full-generation validation passed."
