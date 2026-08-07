# HunyuanOCR-ncnn

HunyuanOCR-1.5 的 CPU-only ncnn 移植、严格精度对齐与
Windows/Linux 双平台部署项目。

## 项目目标

- 使用 pnnx 转换 HunyuanOCR-1.5
- 使用 C++17 实现端到端 OCR 推理
- 部署运行时尽量只依赖 ncnn 和 C++ 标准库
- PyTorch 与 ncnn 最终输出文本严格对齐
- 支持 Ubuntu 24.04 与 Windows x64
- 提供低内存分模块转换和自动对拍工具

## 当前进度

- [x] Ubuntu 24.04 WSL2 环境
- [x] ncnn CPU 构建
- [x] PyTorch CPU 环境
- [x] pnnx 安装
- [x] 最小模型完整转换链路
- [ ] HunyuanOCR 模型下载与版本固定
- [ ] PyTorch CPU 确定性基准
- [ ] Vision 模块转换
- [ ] Text embedding 转换
- [ ] Decoder 与 KV-cache 转换
- [ ] LM head 转换
- [ ] C++ 图像预处理
- [ ] C++ tokenizer
- [ ] C++ 自回归生成
- [ ] Linux 真实模型验证
- [ ] Windows 真实模型验证

## 基线配置

- HunyuanOCR: 1.5 base AR
- Runtime: ncnn CPU
- Reference: PyTorch CPU FP32
- Decoding: greedy
- Build: CMake
- Platforms: Ubuntu 24.04 / Windows x64

## Quick Start

The source code, runtime library, CLI, CMake packaging, tests, and project
documentation are licensed under Apache-2.0. HunyuanOCR model files are not
covered by Apache-2.0; review the Tencent Hunyuan Community License Agreement
before downloading, converting, distributing, or using the model.

Binary packages contain only the runtime, CLI, headers, CMake package, README,
NOTICE, and license files. The converted model directory remains external and
is verified at runtime by `runtime_manifest.tsv` and, when present,
`runtime_compatibility.tsv`.

```bash
# Linux package install
mkdir -p "$HOME/opt/hunyuanocr-ncnn"
tar -xzf HunyuanOCR-ncnn-0.1.0-Linux-x86_64.tar.gz \
  -C "$HOME/opt/hunyuanocr-ncnn" \
  --strip-components=1

"$HOME/opt/hunyuanocr-ncnn/bin/hunyuanocr_cli" \
  --model-dir /path/to/artifacts \
  --image /path/to/receipt.jpg \
  --packing 0 \
  --threads 9 \
  --decoder-cache-mib 512 \
  --max-new-tokens 256 \
  --verify size
```

```powershell
# Windows package install
Expand-Archive `
  -LiteralPath HunyuanOCR-ncnn-0.1.0-Windows-AMD64.zip `
  -DestinationPath D:\opt\hunyuanocr-ncnn

& D:\opt\hunyuanocr-ncnn\HunyuanOCR-ncnn-0.1.0-Windows-AMD64\bin\hunyuanocr_cli.exe `
  --model-dir D:\path\to\model-ntfs `
  --image D:\path\to\receipt.jpg `
  --packing 0 `
  --threads 9 `
  --decoder-cache-mib 512 `
  --max-new-tokens 256 `
  --verify size
```

Detailed user installation guides are in `docs/install_linux.md` and
`docs/install_windows.md`.

## Phase 4A Runtime

The repository now has a reusable C++17 runtime library and OCR CLI. The
runtime accepts the converted model directory and a PNG image, and does not
load any captured PyTorch reference tensor.

```bash
cmake -S . -B build-runtime -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$HOME/.local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn"
cmake --build build-runtime --parallel

build-runtime/hunyuanocr_cli \
  --model-dir "$PWD/artifacts" \
  --image "$PWD/tests/assets/ocr_smoke_en.png" \
  --packing 1 \
  --threads 9 \
  --verify size
```

Generate or refresh `artifacts/runtime_manifest.tsv` after conversion with:

```bash
$HOME/work/hunyuanocr/.venv-reference/bin/python \
  tools/export/export_runtime_manifest.py
```

`--verify size` checks the complete 170-file inventory at normal startup.
`--verify sha256` performs a full content hash check for installation or
release validation. The OCR prompt remains fixed, while image preprocessing,
vision patch grids, image-token spans, multimodal positions, prefill lengths,
and KV caches are now dynamic. Reference-backed parity tests remain available
through `-DHUNYUANOCR_BUILD_PARITY_TESTS=ON`.

## Phase 4B Native Windows

The same runtime now builds and runs with native Windows x64 and MSVC. The
installed package exports `HunyuanOCR::runtime`, and packed/unpacked execution
both reproduce the exact 11-token smoke result through EOS. Reproducible
PowerShell entry points are available in `tools/windows/`; detailed environment,
manifest, performance, memory, and dependency evidence is recorded in
`docs/windows_msvc_milestone.md` and `docs/windows_msvc_validation.json`.

## Phase 4C Dynamic Image Grids

The runtime accepts processor-compatible PNG aspect ratios without a fixed
`[1,22,50]` grid. Linux and native Windows regressions cover wide `[1,16,64]`,
square `[1,32,32]`, tall `[1,48,24]`, and the original smoke grid in both
packed and unpacked modes. See `docs/phase4c_dynamic_image_grid_milestone.md`
for exact tokens, NTFS model-copy measurements, and remaining resource limits.

## Phase 4D Production Inputs And Memory

The runtime now accepts PNG and JPEG input, retains reusable vision position,
merger, and prompt resources across calls, and streams decoder layers by
default. Streaming keeps repeated-process memory near 1.1 GiB instead of
retaining every decoder network. Use `--cache-decode 1` only when the additional
memory is acceptable, and set the workload guard with
`--max-vision-patches N` (default: 2048).

```bash
build-phase4d-optimized/hunyuanocr_cli \
  --model-dir "$PWD/artifacts" \
  --image "$PWD/tests/assets/ocr_receipt_real.jpg" \
  --packing 1 \
  --max-vision-patches 2048 \
  --cache-decode 0
```

The optional repeated-runtime benchmark is enabled with
`-DHUNYUANOCR_BUILD_BENCHMARKS=ON`. Linux and native Windows regressions cover
ten mixed-size calls in one process, exact PyTorch token parity for a real OCR
receipt through its lossless PNG fixture, JPEG decoding compatibility, and
explicit rejection of excessive vision grids. JPEG pixels can differ between
stb_image and Pillow/libjpeg, so exact cross-decoder token parity is asserted on
the lossless PNG fixture. Full evidence and remaining risks are documented in
`docs/phase4d_runtime_memory_milestone.md`.

## Phase 4E Budgeted Decode Cache And JPEG Parity

`--decoder-cache-mib N` retains up to `N` MiB of raw ncnn decoder model bytes.
Layers covered by the budget load from memory on every generation step; the
remaining layers load from files. This avoids the several-fold expansion of
the legacy `--cache-decode 1` mode while reducing long-output latency.

```bash
build-phase4e/hunyuanocr_cli \
  --model-dir "$PWD/artifacts" \
  --image "$PWD/tests/assets/ocr_receipt_real.jpg" \
  --packing 0 \
  --max-new-tokens 256 \
  --decoder-cache-mib 512
```

On the current 24-layer model, 512 MiB covers 7 layers (463 MiB actual model
bytes), while 2048 MiB covers all 24 layers (1585 MiB). The budget limits
retained decoder bytes, not total process RSS; vision tensors, KV cache, the
active decoder layer, and other runtime resources remain additional.

JPEG parity uses a shared pixel contract. `hunyuanocr_decode_image_rgb` exports
the exact stb_image RGB pixels consumed by production, and the PyTorch reference
loads those pixels losslessly. Linux and Windows exports are byte-identical, and
the original JPEG then reproduces all 179 reference tokens through EOS in both
packed and unpacked modes. See
`docs/phase4e_decoder_cache_jpeg_milestone.md` for measurements and semantics.

## Phase 4F Release Tests And Packaging

Release-grade OCR validation is now registered with CTest behind the explicit
`-DHUNYUANOCR_ENABLE_RELEASE_TESTS=ON` option. The registered suites cover the
smoke image, dynamic wide/square/tall images, real PNG, real JPEG, decoder cache
budgets, and negative/error paths. Logs are written to the configured persistent
`HUNYUANOCR_RELEASE_LOG_DIR`.

```bash
cmake -S . -B build-phase4f \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$HOME/.local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn" \
  -DHUNYUANOCR_ENABLE_RELEASE_TESTS=ON \
  -DHUNYUANOCR_RELEASE_MODEL_DIR="$PWD/artifacts" \
  -DHUNYUANOCR_RELEASE_LOG_DIR="$HOME/hunyuanocr-recovery/phase4f/ctest"
cmake --build build-phase4f --parallel
ctest --test-dir build-phase4f --output-on-failure
```

`tools/release/validate_release.py` performs the Linux offline release
acceptance: manifest and compatibility metadata refresh, build, install,
CTest, `find_package(HunyuanOCR)` consumer validation, CPack TGZ/ZIP creation,
and dependency/license audit. Native Windows/MSVC uses
`tools/windows/validate_phase4f_msvc.ps1` and writes the same evidence under
`D:\hunyuanocr-recovery\phase4f`.

The model directory may now include `runtime_compatibility.tsv`. When present,
the runtime verifies the model id, manifest format, file count, runtime version
range, ABI major, precision metadata, and JPEG pixel contract before model
loading. Older model directories without the compatibility file remain accepted
for backward compatibility.

See `docs/phase4f_release_testing_milestone.md` for package hashes, CTest
results, and remaining release risks.

## Phase 4G Open Source Release Preparation

The repository now has a top-level Apache-2.0 `LICENSE`, a release `NOTICE`,
`THIRD_PARTY_NOTICES.md`, archived local license copies for ncnn, stb_image,
and Tencent HunyuanOCR, plus Linux and Windows installation guides. CPack
installs these release documents into the binary packages.

Phase 4G also performs clean package rehearsals: extract the Linux TGZ and
Windows ZIP into fresh directories, verify that release notices are present,
and run OCR from the extracted CLI against the external model directory. Both
platforms reproduce `HELLO 2026\nNCNN CPU TEST` from the package-installed CLI.

See `docs/phase4g_open_source_release_milestone.md` for final package hashes,
license status, dry-run measurements, and remaining release risks.
