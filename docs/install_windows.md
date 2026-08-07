# Windows Installation

This guide covers the native Windows x64 package produced by Phase 4G. The
package contains the CLI, static runtime library, headers, exported CMake
package, README, NOTICE, and license files. It does not contain HunyuanOCR model
weights.

## Requirements

- Windows x64
- Microsoft Visual C++ runtime libraries
- OpenMP runtime used by the MSVC/ncnn build
- Converted HunyuanOCR-ncnn model directory with `runtime_manifest.tsv`
- Optional but recommended: `runtime_compatibility.tsv`

The validated build used MSVC 19.51 and CPU-only ncnn.

## Model License

The model files are governed by the Tencent Hunyuan Community License
Agreement, not by this repository's Apache-2.0 source-code license. Review the
license in `third_party\licenses\Tencent-HunyuanOCR-LICENSE.txt` before
downloading, converting, distributing, or using the model.

## Install The Binary Package

Extract the package to an application directory:

```powershell
Expand-Archive `
  -LiteralPath HunyuanOCR-ncnn-0.1.0-Windows-AMD64.zip `
  -DestinationPath D:\opt\hunyuanocr-ncnn
```

The installed CLI will be:

```text
D:\opt\hunyuanocr-ncnn\HunyuanOCR-ncnn-0.1.0-Windows-AMD64\bin\hunyuanocr_cli.exe
```

Release notices are installed under:

```text
D:\opt\hunyuanocr-ncnn\HunyuanOCR-ncnn-0.1.0-Windows-AMD64\share\doc\HunyuanOCR_ncnn\
```

## Verify The Model Directory

Run a size check for normal startup:

```powershell
& D:\opt\hunyuanocr-ncnn\HunyuanOCR-ncnn-0.1.0-Windows-AMD64\bin\hunyuanocr_cli.exe `
  --model-dir D:\path\to\model-ntfs `
  --image D:\path\to\image.png `
  --verify size
```

For release or archival validation, use the full SHA-256 manifest check:

```powershell
& D:\opt\hunyuanocr-ncnn\HunyuanOCR-ncnn-0.1.0-Windows-AMD64\bin\hunyuanocr_cli.exe `
  --model-dir D:\path\to\model-ntfs `
  --image D:\path\to\image.png `
  --verify sha256
```

## Run OCR

```powershell
& D:\opt\hunyuanocr-ncnn\HunyuanOCR-ncnn-0.1.0-Windows-AMD64\bin\hunyuanocr_cli.exe `
  --model-dir D:\path\to\model-ntfs `
  --image D:\path\to\receipt.jpg `
  --packing 0 `
  --threads 9 `
  --decoder-cache-mib 512 `
  --max-new-tokens 256 `
  --verify size
```

Use `--packing 0` as the default FP32 CPU path. Packed mode remains available
with `--packing 1` for compatibility testing.

## CMake Consumer

If ncnn is also available through `find_package(ncnn CONFIG)`, consumers can use
the installed runtime package:

```powershell
cmake -S your_app -B build `
  -DCMAKE_PREFIX_PATH="D:\opt\hunyuanocr-ncnn\HunyuanOCR-ncnn-0.1.0-Windows-AMD64;D:\path\to\ncnn-install"
cmake --build build --config Release
```

Link against:

```cmake
target_link_libraries(your_app PRIVATE HunyuanOCR::runtime)
```

## Offline Release Rehearsal

The Phase 4G dry run extracted the ZIP package into a clean directory and
executed the installed CLI against the external model directory. The recorded
report is `docs/windows_phase4g_release_dryrun.json`.
