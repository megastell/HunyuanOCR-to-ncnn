#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${HUNYUANOCR_PHASE4K_BUILD_DIR:-$ROOT/build-phase4k}"
NCNN_DIR="${HUNYUANOCR_NCNN_DIR:-$HOME/.local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn}"
HF_MODEL_DIR="${HUNYUANOCR_HF_MODEL_DIR:-$HOME/work/hunyuanocr/models/HunyuanOCR-1.5}"
WORK_DIR="${HUNYUANOCR_PHASE4K_WORK_DIR:-$HOME/hunyuanocr-recovery/phase4k}"
REFERENCE_PYTHON="${HUNYUANOCR_REFERENCE_PYTHON:-$HOME/work/hunyuanocr/.venv-reference/bin/python}"
PNNX="${HUNYUANOCR_PNNX:-$HOME/work/hunyuanocr/.venv-pnnx/bin/pnnx}"

cd "$ROOT"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$NCNN_DIR" \
  -DHUNYUANOCR_ENABLE_REPRODUCTION_TESTS=ON \
  -DHUNYUANOCR_REPRO_HF_MODEL_DIR="$HF_MODEL_DIR" \
  -DHUNYUANOCR_REPRO_WORK_DIR="$WORK_DIR" \
  -DHUNYUANOCR_REPRO_REFERENCE_PYTHON="$REFERENCE_PYTHON" \
  -DHUNYUANOCR_REPRO_PNNX="$PNNX"

cmake --build "$BUILD_DIR" --parallel

ctest \
  --test-dir "$BUILD_DIR" \
  -L reproducible-artifacts \
  --output-on-failure
