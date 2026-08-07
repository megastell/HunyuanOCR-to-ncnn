#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${HUNYUANOCR_PHASE4L_WORK_DIR:-$HOME/hunyuanocr-recovery/phase4l/linux}"
MODEL_DIR="${HUNYUANOCR_PHASE4L_MODEL_DIR:-$HOME/hunyuanocr-recovery/phase4k/direct-staging-artifacts}"
NCNN_DIR="${HUNYUANOCR_NCNN_DIR:-$HOME/.local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn}"
PYTHON="${HUNYUANOCR_REFERENCE_PYTHON:-$HOME/work/hunyuanocr/.venv-reference/bin/python}"

cd "$ROOT"
mkdir -p "$WORK_DIR"

"$PYTHON" tools/release/validate_release.py \
  --phase 4L \
  --model-dir "$MODEL_DIR" \
  --ncnn-dir "$NCNN_DIR" \
  --build-dir "$WORK_DIR/build" \
  --install-dir "$WORK_DIR/install" \
  --package-dir "$WORK_DIR/packages" \
  --log-dir "$WORK_DIR/logs" \
  --report "$ROOT/docs/linux_phase4l_release_validation.json" \
  --skip-manifest-refresh

PACKAGE="$(find "$WORK_DIR/packages" -maxdepth 1 -type f -name 'HunyuanOCR-ncnn-*.tar.gz' | sort | tail -1)"
if [ -z "$PACKAGE" ]; then
  echo "No Linux TGZ package was produced in $WORK_DIR/packages" >&2
  exit 1
fi

for packing in 0 1; do
  "$PYTHON" tools/release/rehearse_binary_package.py \
    --package "$PACKAGE" \
    --model-dir "$MODEL_DIR" \
    --work-dir "$WORK_DIR/package-rehearsal-packing${packing}" \
    --log-dir "$WORK_DIR/logs/package-rehearsal-packing${packing}" \
    --report "$ROOT/docs/linux_phase4l_package_rehearsal_packing${packing}.json" \
    --packing "$packing"
done
