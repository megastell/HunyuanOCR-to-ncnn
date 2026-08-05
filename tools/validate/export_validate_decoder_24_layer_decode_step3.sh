#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/work/hunyuanocr/HunyuanOCR-ncnn"

REFERENCE_PYTHON="$HOME/work/hunyuanocr/.venv-reference/bin/python"
PNNX="$HOME/work/hunyuanocr/.venv-pnnx/bin/pnnx"

NCNN_PREFIX="$HOME/.local/ncnn-cpu-ropefix-rmsnorm"

EXPORTER="$ROOT/tools/export/export_decoder_decode_step3.py"

PNNX_VALIDATOR="$ROOT/tools/validate/validate_decoder_layer0_decode_step3_pnnx.py"
NCNN_RUNNER="$ROOT/tools/validate/validate_decoder_24_layer_decode_step3_ncnn.sh"

CPP_SOURCE="$ROOT/tests/decoder_layer0_decode_step3/cpp"
CPP_BUILD="$ROOT/tests/decoder_layer0_decode_step3/build-ropefix-rmsnorm"

CPP_BIN="$CPP_BUILD/decoder_layer0_decode_step3_parity"

EXPORT_SUMMARY="$ROOT/docs/decoder_24_layer_decode_step3_export_summary.txt"
PNNX_SUMMARY="$ROOT/docs/decoder_24_layer_decode_step3_pnnx_summary.txt"
BATCH_LOG="$ROOT/docs/decoder_24_layer_decode_step3_ncnn_batch.txt"

cd "$ROOT"

echo "============================================================"
echo "Step 3 fixed-shape contract"
echo "============================================================"
echo "mask length    : 316"
echo "past KV length : 315"
echo "present length : 316"
echo

for executable in \
  "$REFERENCE_PYTHON" \
  "$PNNX"
do
  if [ ! -x "$executable" ]; then
    echo "❌ 可执行文件不存在：$executable"
    exit 1
  fi
done

for source in \
  "$EXPORTER" \
  "$PNNX_VALIDATOR" \
  "$NCNN_RUNNER" \
  "$CPP_SOURCE/main.cpp" \
  "$CPP_SOURCE/CMakeLists.txt"
do
  if [ ! -s "$source" ]; then
    echo "❌ 源文件不存在或为空：$source"
    exit 1
  fi
done

for layer in $(seq 0 23)
do
  REF_DIR="$ROOT/reference/smoke_en_cpu_fp32/decoder_layer${layer}_decode_step3"

  for file in \
    "layer${layer}_hidden_states.npy" \
    "layer${layer}_attention_mask.npy" \
    "layer${layer}_position_embeddings_0.npy" \
    "layer${layer}_position_embeddings_1.npy" \
    "layer${layer}_output.npy" \
    "past_key.npy" \
    "past_value.npy" \
    "present_key.npy" \
    "present_value.npy"
  do
    if [ ! -s "$REF_DIR/$file" ]; then
      echo "❌ Step 3参考张量缺失：$REF_DIR/$file"
      exit 1
    fi
  done
done

echo "✅ 全部24层Step 3 NPY参考张量存在"

echo
echo "============================================================"
echo "1. Export 24 TorchScript decoder layers"
echo "============================================================"

: > "$EXPORT_SUMMARY"

for layer in $(seq 0 23)
do
  NAME="decoder_layer${layer}_decode_step3"
  MODEL_DIR="$ROOT/artifacts/$NAME"

  LOG="$ROOT/docs/${NAME}_export.txt"
  REPORT="$ROOT/docs/${NAME}_export.json"

  mkdir -p "$MODEL_DIR"

  rm -f \
    "$MODEL_DIR/${NAME}.pt" \
    "$MODEL_DIR/${NAME}.pnnx.param" \
    "$MODEL_DIR/${NAME}.pnnx.bin" \
    "$MODEL_DIR/${NAME}.pnnx.onnx" \
    "$MODEL_DIR/${NAME}.ncnn.param" \
    "$MODEL_DIR/${NAME}.ncnn.bin" \
    "$MODEL_DIR/${NAME}_pnnx.py" \
    "$MODEL_DIR/${NAME}_ncnn.py" \
    "$LOG" \
    "$REPORT"

  START_SECONDS=$SECONDS

  if "$REFERENCE_PYTHON" \
    "$EXPORTER" \
    --layer-index "$layer" \
    > "$LOG" \
    2>&1
  then
    STATUS=0
  else
    STATUS=$?
  fi

  ELAPSED=$((SECONDS - START_SECONDS))

  printf \
    'Layer %02d status=%d elapsed=%ds\n' \
    "$layer" \
    "$STATUS" \
    "$ELAPSED" |
    tee -a "$EXPORT_SUMMARY"

  if [ "$STATUS" -ne 0 ]; then
    echo "❌ Layer ${layer} TorchScript导出失败"
    tail -120 "$LOG"
    exit "$STATUS"
  fi

  if [ ! -s "$MODEL_DIR/${NAME}.pt" ]; then
    echo "❌ Layer ${layer} TorchScript文件缺失"
    exit 1
  fi
done

echo "✅ 全部24层Step 3 TorchScript导出成功"

echo
echo "============================================================"
echo "2. Convert all 24 layers with PNNX"
echo "============================================================"

: > "$PNNX_SUMMARY"

INPUT_SHAPE='inputshape=[1,1,1024]f32,[1,1,1,316]f32,[4,1,1,128]f32,[4,1,1,128]f32,[1,8,315,128]f32,[1,8,315,128]f32'

for layer in $(seq 0 23)
do
  NAME="decoder_layer${layer}_decode_step3"
  MODEL_DIR="$ROOT/artifacts/$NAME"
  LOG="$ROOT/docs/${NAME}_pnnx.txt"

  START_SECONDS=$SECONDS

  if (
    cd "$MODEL_DIR"

    /usr/bin/time -v \
      "$PNNX" \
      "./${NAME}.pt" \
      "$INPUT_SHAPE" \
      fp16=0 \
      optlevel=2 \
      device=cpu
  ) > "$LOG" 2>&1
  then
    STATUS=0
  else
    STATUS=$?
  fi

  ELAPSED=$((SECONDS - START_SECONDS))

  printf \
    'Layer %02d status=%d elapsed=%ds\n' \
    "$layer" \
    "$STATUS" \
    "$ELAPSED" |
    tee -a "$PNNX_SUMMARY"

  if [ "$STATUS" -ne 0 ]; then
    echo "❌ Layer ${layer} PNNX转换失败"
    tail -150 "$LOG"
    exit "$STATUS"
  fi

  for file in \
    "${NAME}.pnnx.param" \
    "${NAME}.pnnx.bin" \
    "${NAME}_pnnx.py" \
    "${NAME}.ncnn.param" \
    "${NAME}.ncnn.bin"
  do
    if [ ! -s "$MODEL_DIR/$file" ]; then
      echo "❌ PNNX产物缺失：$MODEL_DIR/$file"
      exit 1
    fi
  done
done

echo "✅ 全部24层Step 3 PNNX/ncnn转换成功"

echo
echo "============================================================"
echo "3. Audit all export reports and fixed graph shapes"
echo "============================================================"

"$REFERENCE_PYTHON" - <<'PY'
import json
from pathlib import Path


root = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

expected_inputs = {
    "hidden_states": [1, 1, 1024],
    "attention_mask": [1, 1, 1, 316],
    "rope_cos": [4, 1, 1, 128],
    "rope_sin": [4, 1, 1, 128],
    "past_key": [1, 8, 315, 128],
    "past_value": [1, 8, 315, 128],
}

expected_outputs = {
    "layer_output": [1, 1, 1024],
    "present_key": [1, 8, 316, 128],
    "present_value": [1, 8, 316, 128],
}

for layer in range(24):
    name = (
        f"decoder_layer{layer}_decode_step3"
    )

    report_path = (
        root
        / "docs"
        / f"{name}_export.json"
    )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert report["layer_index"] == layer
    assert report["decode_name"] == name

    assert (
        report["contract"]["inputs"]
        == expected_inputs
    )

    assert (
        report["contract"]["outputs"]
        == expected_outputs
    )

    for metric_group in (
        "eager_metrics",
        "torchscript_metrics",
    ):
        for output_name in (
            "layer_output",
            "present_key",
            "present_value",
        ):
            metrics = (
                report[metric_group][output_name]
            )

            assert (
                metrics["maximum_abs_error"]
                == 0.0
            )

            assert (
                metrics["mean_abs_error"]
                == 0.0
            )

    model_dir = (
        root
        / "artifacts"
        / name
    )

    pnnx_text = (
        model_dir
        / f"{name}.pnnx.param"
    ).read_text(
        encoding="utf-8"
    )

    required_shapes = (
        "(1,1,1,316)f32",
        "(1,8,315,128)f32",
        "(1,8,316,128)f32",
    )

    for shape in required_shapes:
        if shape not in pnnx_text:
            raise RuntimeError(
                f"{name}缺少PNNX形状：{shape}"
            )

    ncnn_lines = (
        model_dir
        / f"{name}.ncnn.param"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    cache_reshape_lines = [
        line
        for line in ncnn_lines
        if (
            line.split()
            and line.split()[0] == "Reshape"
            and "1=316" in line
        )
    ]

    if len(cache_reshape_lines) < 2:
        raise RuntimeError(
            f"{name}的ncnn缓存Reshape不是316："
            f"{cache_reshape_lines}"
        )

print(
    "✅ 24层导出报告、PNNX形状及"
    "ncnn固定缓存长度全部正确"
)
PY

echo
echo "============================================================"
echo "4. Validate Layer 0 generated PNNX Python model"
echo "============================================================"

PNNX_PYTHON_LOG="$ROOT/docs/decoder_layer0_decode_step3_pnnx_python.txt"

"$REFERENCE_PYTHON" \
  "$PNNX_VALIDATOR" \
  > "$PNNX_PYTHON_LOG" \
  2>&1

grep -E \
  'maximum abs error|mean abs error|cosine similarity|数值对齐成功|❌' \
  "$PNNX_PYTHON_LOG"

echo "✅ Layer 0 Step 3 PNNX Python parity通过"

echo
echo "============================================================"
echo "5. Build Step 3 ncnn C++ validator"
echo "============================================================"

rm -rf "$CPP_BUILD"

cmake \
  -S "$CPP_SOURCE" \
  -B "$CPP_BUILD" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$NCNN_PREFIX" \
  > "$ROOT/docs/decoder_layer0_decode_step3_cmake_configure.txt" \
  2>&1

cmake \
  --build "$CPP_BUILD" \
  --parallel \
  > "$ROOT/docs/decoder_layer0_decode_step3_cmake_build.txt" \
  2>&1

if [ ! -x "$CPP_BIN" ]; then
  echo "❌ Step 3 C++验证器没有生成"
  exit 1
fi

echo "✅ Step 3 C++验证器构建成功"

echo
echo "============================================================"
echo "6. Validate Layer 0 unpacked and packed"
echo "============================================================"

NAME="decoder_layer0_decode_step3"
MODEL_DIR="$ROOT/artifacts/$NAME"
REF_DIR="$MODEL_DIR/reference"

ARGS=(
  "$MODEL_DIR/${NAME}.ncnn.param"
  "$MODEL_DIR/${NAME}.ncnn.bin"
  "$REF_DIR/layer0_hidden_states_f32.bin"
  "$REF_DIR/layer0_attention_mask_f32.bin"
  "$REF_DIR/layer0_position_embeddings_0_f32.bin"
  "$REF_DIR/layer0_position_embeddings_1_f32.bin"
  "$REF_DIR/past_key_f32.bin"
  "$REF_DIR/past_value_f32.bin"
  "$REF_DIR/layer0_output_f32.bin"
  "$REF_DIR/present_key_f32.bin"
  "$REF_DIR/present_value_f32.bin"
)

"$CPP_BIN" \
  "${ARGS[@]}" \
  0 \
  > "$ROOT/docs/decoder_layer0_decode_step3_ncnn_cpp_unpacked.txt" \
  2>&1

"$CPP_BIN" \
  "${ARGS[@]}" \
  1 \
  > "$ROOT/docs/decoder_layer0_decode_step3_ncnn_cpp_packed.txt" \
  2>&1

grep -E \
  'out0 layer_output|out1 present_key|out2 present_value|Maximum abs error|history prefix|max error|数值对齐成功|❌' \
  "$ROOT/docs/decoder_layer0_decode_step3_ncnn_cpp_unpacked.txt"

echo
echo "----- packed -----"

grep -E \
  'out0 layer_output|out1 present_key|out2 present_value|Maximum abs error|history prefix|max error|数值对齐成功|❌' \
  "$ROOT/docs/decoder_layer0_decode_step3_ncnn_cpp_packed.txt"

echo "✅ Layer 0 packed和unpacked均通过"

echo
echo "============================================================"
echo "7. Validate all 24 packed ncnn graphs"
echo "============================================================"

"$NCNN_RUNNER" \
  > "$BATCH_LOG" \
  2>&1

cat \
  "$ROOT/docs/decoder_24_layer_decode_step3_ncnn_summary.txt"

if grep -qv \
  'status=0' \
  "$ROOT/docs/decoder_24_layer_decode_step3_ncnn_summary.txt"
then
  echo "❌ 至少一层Step 3 ncnn状态非0"
  exit 1
fi

if grep -R \
  'history prefix max error' \
  "$ROOT"/docs/decoder_layer*_decode_step3_ncnn_cpp_packed.txt |
  grep -v \
  '0.0000000000e+00'
then
  echo "❌ 至少一层KV历史前缀发生变化"
  exit 1
fi

echo "✅ 24层packed ncnn全部通过且KV前缀误差为0"

echo
echo "============================================================"
echo "8. Clean logs and generate aggregate parity report"
echo "============================================================"

"$REFERENCE_PYTHON" - <<'PY'
from pathlib import Path


patterns = (
    "decoder_layer*_decode_step3_export.txt",
    "decoder_layer*_decode_step3_pnnx.txt",
    "decoder_layer*_decode_step3_ncnn_cpp_*.txt",
    "decoder_24_layer_decode_step3_*.txt",
    "decoder_layer0_decode_step3_cmake_*.txt",
)

paths: set[Path] = set()

for pattern in patterns:
    paths.update(
        Path("docs").glob(pattern)
    )

for path in sorted(paths):
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).replace("\r", "\n")

    cleaned: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if "Loading weights:" in line:
            continue

        if (
            line == ""
            and cleaned
            and cleaned[-1] == ""
        ):
            continue

        cleaned.append(line)

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    path.write_text(
        "\n".join(cleaned) + "\n",
        encoding="utf-8",
    )

print(
    f"✅ 已清理{len(paths)}份Step 3日志"
)
PY

"$REFERENCE_PYTHON" - <<'PY'
from __future__ import annotations

import json
import re
from pathlib import Path


root = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

docs = root / "docs"

number = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"


def extract_metric(
    text: str,
    label: str,
) -> dict[str, float]:
    pattern = re.compile(
        re.escape(label)
        + r"\s*\n"
        + r"\s*Maximum abs error\s*:\s*("
        + number
        + r")\s*\n"
        + r"\s*Mean abs error\s*:\s*("
        + number
        + r")",
    )

    match = pattern.search(text)

    if match is None:
        raise RuntimeError(
            f"无法解析指标：{label}"
        )

    return {
        "maximum_abs_error":
            float(match.group(1)),
        "mean_abs_error":
            float(match.group(2)),
    }


def extract_scalar(
    text: str,
    label: str,
) -> float:
    pattern = re.compile(
        re.escape(label)
        + r"\s*:\s*("
        + number
        + r")"
    )

    match = pattern.search(text)

    if match is None:
        raise RuntimeError(
            f"无法解析标量：{label}"
        )

    return float(match.group(1))


layers = []

maximum = {
    "layer_output": 0.0,
    "present_key": 0.0,
    "present_value": 0.0,
    "key_history_prefix": 0.0,
    "key_appended_token": 0.0,
    "value_history_prefix": 0.0,
    "value_appended_token": 0.0,
}

for layer in range(24):
    log_path = (
        docs
        / (
            f"decoder_layer{layer}_"
            "decode_step3_ncnn_cpp_packed.txt"
        )
    )

    text = log_path.read_text(
        encoding="utf-8"
    )

    output = extract_metric(
        text,
        "out0 layer_output",
    )

    key = extract_metric(
        text,
        "out1 present_key",
    )

    value = extract_metric(
        text,
        "out2 present_value",
    )

    key_prefix = extract_scalar(
        text,
        "Key history prefix max error",
    )

    key_new = extract_scalar(
        text,
        "Key appended token max error",
    )

    value_prefix = extract_scalar(
        text,
        "Value history prefix max error",
    )

    value_new = extract_scalar(
        text,
        "Value appended token max error",
    )

    layer_report = {
        "layer_index": layer,
        "layer_output": output,
        "present_key": key,
        "present_value": value,
        "key_history_prefix_max_error":
            key_prefix,
        "key_appended_token_max_error":
            key_new,
        "value_history_prefix_max_error":
            value_prefix,
        "value_appended_token_max_error":
            value_new,
    }

    layers.append(layer_report)

    maximum["layer_output"] = max(
        maximum["layer_output"],
        output["maximum_abs_error"],
    )

    maximum["present_key"] = max(
        maximum["present_key"],
        key["maximum_abs_error"],
    )

    maximum["present_value"] = max(
        maximum["present_value"],
        value["maximum_abs_error"],
    )

    maximum["key_history_prefix"] = max(
        maximum["key_history_prefix"],
        key_prefix,
    )

    maximum["key_appended_token"] = max(
        maximum["key_appended_token"],
        key_new,
    )

    maximum["value_history_prefix"] = max(
        maximum["value_history_prefix"],
        value_prefix,
    )

    maximum["value_appended_token"] = max(
        maximum["value_appended_token"],
        value_new,
    )


report = {
    "decode_step": 3,
    "layer_count": 24,
    "threads": 9,
    "packing_layout": True,
    "vulkan": False,
    "past_length": 315,
    "present_length": 316,
    "token_contract": {
        "input_token": 206,
        "output_token": 1717,
    },
    "maximum_errors": maximum,
    "layers": layers,
}

report_path = (
    docs
    / "decoder_24_layer_decode_step3_ncnn_parity.json"
)

report_path.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

milestone = f"""# Decoder 24-layer Decode Step 3 ncnn Milestone

## Scope

This milestone validates the third fixed-shape single-token
decode graph for all 24 HunyuanOCR-1.5 decoder layers.

## Token contract

- Step 3 input token: 206
- Step 3 output token: 1717

## Shape contract

- current-token length: 1
- attention-mask length: 316
- past-KV length: 315
- present-KV length: 316
- KV heads: 8
- head dimension: 128

## Export pipeline

All 24 layers passed:

- PyTorch eager reference parity
- TorchScript trace parity
- PNNX conversion
- fixed-shape graph audit
- packed ncnn C++ parity

Layer 0 additionally passed unpacked ncnn execution.

## Maximum ncnn errors

- layer output:
  {maximum["layer_output"]:.10e}
- present Key:
  {maximum["present_key"]:.10e}
- present Value:
  {maximum["present_value"]:.10e}
- Key history prefix:
  {maximum["key_history_prefix"]:.10e}
- Key appended token:
  {maximum["key_appended_token"]:.10e}
- Value history prefix:
  {maximum["value_history_prefix"]:.10e}
- Value appended token:
  {maximum["value_appended_token"]:.10e}

## Result

All same-layer 315-token KV histories remain unchanged while
the 316th token is appended.

This milestone validates fixed-shape Step 3 execution. It does
not yet constitute a dynamic-length generation loop.
"""

(
    docs
    / "decoder_24_layer_decode_step3_ncnn_milestone.md"
).write_text(
    milestone,
    encoding="utf-8",
)

print(
    "Maximum errors:",
    json.dumps(
        maximum,
        ensure_ascii=False,
        indent=2,
    ),
)

print(
    "✅ Step 3汇总JSON和里程碑文档生成成功"
)
PY

git diff --check

echo
echo "============================================================"
echo "✅ Step 3 24-layer export and ncnn validation completed"
echo "============================================================"

echo
echo "Export summary:"
cat "$EXPORT_SUMMARY"

echo
echo "PNNX summary:"
cat "$PNNX_SUMMARY"

echo
echo "ncnn summary:"
cat "$ROOT/docs/decoder_24_layer_decode_step3_ncnn_summary.txt"

echo
echo "Working tree:"
git status --short
