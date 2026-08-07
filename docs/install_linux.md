# Linux Installation

This guide covers the binary runtime package produced by Phase 4G. The package
contains the C++ runtime, CLI, headers, exported CMake package, README, NOTICE,
and license files. It does not contain HunyuanOCR model weights.

## Requirements

- Linux x86_64
- CPU runtime with OpenMP-compatible libraries available
- Converted HunyuanOCR-ncnn model directory with `runtime_manifest.tsv`
- Optional but recommended: `runtime_compatibility.tsv`

The validated ncnn build is CPU-only with the local RoPE/RMSNorm precision
changes used by this project.

## Model License

The model files are governed by the Tencent Hunyuan Community License
Agreement, not by this repository's Apache-2.0 source-code license. Review the
license in `third_party/licenses/Tencent-HunyuanOCR-LICENSE.txt` before
downloading, converting, distributing, or using the model.

## Install The Binary Package

```bash
mkdir -p "$HOME/opt/hunyuanocr-ncnn"
tar -xzf HunyuanOCR-ncnn-0.1.0-Linux-x86_64.tar.gz \
  -C "$HOME/opt/hunyuanocr-ncnn" \
  --strip-components=1
```

The installed CLI will be:

```text
$HOME/opt/hunyuanocr-ncnn/bin/hunyuanocr_cli
```

Release notices are installed under:

```text
$HOME/opt/hunyuanocr-ncnn/share/doc/HunyuanOCR_ncnn/
```

## Verify The Model Directory

Run a size check for normal startup:

```bash
"$HOME/opt/hunyuanocr-ncnn/bin/hunyuanocr_cli" \
  --model-dir /path/to/artifacts \
  --image /path/to/image.png \
  --verify size
```

For release or archival validation, use the full SHA-256 manifest check:

```bash
"$HOME/opt/hunyuanocr-ncnn/bin/hunyuanocr_cli" \
  --model-dir /path/to/artifacts \
  --image /path/to/image.png \
  --verify sha256
```

## Run OCR

```bash
"$HOME/opt/hunyuanocr-ncnn/bin/hunyuanocr_cli" \
  --model-dir /path/to/artifacts \
  --image /path/to/receipt.jpg \
  --packing 0 \
  --threads 9 \
  --decoder-cache-mib 512 \
  --max-new-tokens 256 \
  --verify size
```

Use `--packing 0` as the default FP32 CPU path. Packed mode remains available
with `--packing 1` for compatibility testing.

## CMake Consumer

If ncnn is also available through `find_package(ncnn CONFIG)`, consumers can use
the installed runtime package:

```bash
cmake -S your_app -B build \
  -DCMAKE_PREFIX_PATH="$HOME/opt/hunyuanocr-ncnn;/path/to/ncnn/install"
cmake --build build
```

Link against:

```cmake
target_link_libraries(your_app PRIVATE HunyuanOCR::runtime)
```

## Offline Release Rehearsal

The Phase 4G dry run extracted the package into a clean directory and executed
the installed CLI against the external model directory. The recorded report is
`docs/linux_phase4g_release_dryrun.json`.

## Reproduce Runtime Artifacts From A Local HF Model

For release acceptance, the runtime artifacts can be rebuilt from an existing
local `tencent/HunyuanOCR` HuggingFace model directory without downloading the
model again. The pipeline regenerates PyTorch reference tensors, runs the
direct pnnx/ncnn export into a clean persistent staging directory, generates
`runtime_manifest.tsv` and `runtime_compatibility.tsv`, then runs the Linux
smoke, dynamic-size, real PNG, real JPEG, and cache-budget validations.

```bash
cmake -S . -B build-phase4k \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$HOME/.local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn" \
  -DHUNYUANOCR_ENABLE_REPRODUCTION_TESTS=ON \
  -DHUNYUANOCR_REPRO_HF_MODEL_DIR="$HOME/work/hunyuanocr/models/HunyuanOCR-1.5" \
  -DHUNYUANOCR_REPRO_WORK_DIR="$HOME/hunyuanocr-recovery/phase4k" \
  -DHUNYUANOCR_REPRO_REFERENCE_PYTHON="$HOME/work/hunyuanocr/.venv-reference/bin/python" \
  -DHUNYUANOCR_REPRO_PNNX="$HOME/work/hunyuanocr/.venv-pnnx/bin/pnnx"
cmake --build build-phase4k --parallel
ctest --test-dir build-phase4k -L reproducible-artifacts --output-on-failure
```

Direct script entry:

```bash
$HOME/work/hunyuanocr/.venv-reference/bin/python \
  tools/release/reproduce_runtime_artifacts_acceptance.py \
  --clean-staging \
  --hf-model-dir "$HOME/work/hunyuanocr/models/HunyuanOCR-1.5" \
  --work-dir "$HOME/hunyuanocr-recovery/phase4k"
```

The staging artifacts are written to
`$HOME/hunyuanocr-recovery/phase4k/direct-staging-artifacts`. The acceptance
report is `docs/phase4k_reproducible_release_acceptance.json`. The script
checks that the existing repository `artifacts/` and `reference/` trees are
unchanged after the run.
