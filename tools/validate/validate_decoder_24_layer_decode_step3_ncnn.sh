#!/usr/bin/env bash

set -uo pipefail

ROOT="$HOME/work/hunyuanocr/HunyuanOCR-ncnn"

BIN="$ROOT/tests/decoder_layer0_decode_step3/build-ropefix-rmsnorm/decoder_layer0_decode_step3_parity"

SUMMARY="$ROOT/docs/decoder_24_layer_decode_step3_ncnn_summary.txt"

if [ ! -x "$BIN" ]; then
  echo "❌ C++验证器不存在：$BIN"
  exit 1
fi

: > "$SUMMARY"

FAIL=0

for layer in $(seq 0 23)
do
  NAME="decoder_layer${layer}_decode_step3"

  MODEL_DIR="$ROOT/artifacts/$NAME"
  REF_DIR="$MODEL_DIR/reference"
  LOG="$ROOT/docs/${NAME}_ncnn_cpp_packed.txt"

  PARAM="$MODEL_DIR/${NAME}.ncnn.param"
  MODEL="$MODEL_DIR/${NAME}.ncnn.bin"

  HIDDEN="$REF_DIR/layer${layer}_hidden_states_f32.bin"
  MASK="$REF_DIR/layer${layer}_attention_mask_f32.bin"
  ROPE_COS="$REF_DIR/layer${layer}_position_embeddings_0_f32.bin"
  ROPE_SIN="$REF_DIR/layer${layer}_position_embeddings_1_f32.bin"

  PAST_KEY="$REF_DIR/past_key_f32.bin"
  PAST_VALUE="$REF_DIR/past_value_f32.bin"

  EXPECTED_OUTPUT="$REF_DIR/layer${layer}_output_f32.bin"
  EXPECTED_KEY="$REF_DIR/present_key_f32.bin"
  EXPECTED_VALUE="$REF_DIR/present_value_f32.bin"

  echo
  echo "============================================================"
  echo "Validate Layer ${layer}/23"
  echo "============================================================"

  REQUIRED_FILES=(
    "$PARAM"
    "$MODEL"
    "$HIDDEN"
    "$MASK"
    "$ROPE_COS"
    "$ROPE_SIN"
    "$PAST_KEY"
    "$PAST_VALUE"
    "$EXPECTED_OUTPUT"
    "$EXPECTED_KEY"
    "$EXPECTED_VALUE"
  )

  MISSING=0

  for file in "${REQUIRED_FILES[@]}"
  do
    if [ ! -s "$file" ]; then
      echo "❌ 缺少文件：$file"
      MISSING=1
    fi
  done

  if [ "$MISSING" -ne 0 ]; then
    printf \
      'Layer %02d status=missing\n' \
      "$layer" |
      tee -a "$SUMMARY"

    FAIL=1
    continue
  fi

  START_SECONDS=$SECONDS

  "$BIN" \
    "$PARAM" \
    "$MODEL" \
    "$HIDDEN" \
    "$MASK" \
    "$ROPE_COS" \
    "$ROPE_SIN" \
    "$PAST_KEY" \
    "$PAST_VALUE" \
    "$EXPECTED_OUTPUT" \
    "$EXPECTED_KEY" \
    "$EXPECTED_VALUE" \
    1 \
    > "$LOG" \
    2>&1

  STATUS=$?

  # 通用C++验证器内部的成功文字固定写成Layer 0。
  # 这里只修正日志中的显示层号，不改变任何数值结果。
  if [ -f "$LOG" ]; then
    sed -i       "s/Decoder Layer 0 Decode Step 3/Decoder Layer ${layer} Decode Step 3/g"       "$LOG"
  fi

  ELAPSED=$((SECONDS - START_SECONDS))

  printf \
    'Layer %02d status=%d elapsed=%ds\n' \
    "$layer" \
    "$STATUS" \
    "$ELAPSED" |
    tee -a "$SUMMARY"

  grep -E \
    'out0 layer_output|out1 present_key|out2 present_value|Maximum abs error|Mean abs error|history prefix|max error|数值对齐成功|❌' \
    "$LOG"

  if [ "$STATUS" -ne 0 ]; then
    echo "❌ Layer ${layer} ncnn验证失败"
    tail -100 "$LOG"
    FAIL=1
  fi
done

echo

if [ "$FAIL" -ne 0 ]; then
  echo "❌ 至少一层Step 3 ncnn数值验证失败"
  exit 1
fi

echo "✅ 全部24层Step 3 packed ncnn独立数值验证成功"
