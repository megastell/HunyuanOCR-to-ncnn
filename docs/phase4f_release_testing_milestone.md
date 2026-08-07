# Phase 4F Release Tests And Packaging

## Scope

Phase 4F turns the existing ad-hoc end-to-end validation into release-grade
automation. It registers the OCR suites with CTest, adds installable binary
packages, introduces model/runtime compatibility metadata, and records offline
release acceptance on Linux and native Windows/MSVC.

## CTest Suites

CTest registration is explicit and opt-in through
`-DHUNYUANOCR_ENABLE_RELEASE_TESTS=ON`. This keeps ordinary local builds fast
while making release validation reproducible.

| Test | Coverage |
| --- | --- |
| `hunyuanocr_smoke_exact` | 11-token `HELLO 2026\nNCNN CPU TEST`, packed and unpacked |
| `hunyuanocr_dynamic_sizes_exact` | wide, square, and tall OCR images, packed and unpacked |
| `hunyuanocr_real_png_exact` | real receipt PNG, 179 tokens through EOS |
| `hunyuanocr_real_jpeg_exact` | real receipt JPEG using the stb RGB pixel contract |
| `hunyuanocr_decoder_cache_budgets` | 0, 512, and 2048 MiB decoder-cache budgets |
| `hunyuanocr_error_paths` | cache-mode conflict, runtime/model version reject, manifest missing file, vision patch limit |

The tests are driven by `tools/validate/validate_runtime_suite.py`. Each suite
writes raw CLI logs and a JSON summary to the configured persistent log
directory.

## Model Compatibility

`tools/export/export_runtime_manifest.py` now writes
`runtime_compatibility.tsv` beside `runtime_manifest.tsv`. The runtime validates
the compatibility file when it is present and accepts older model directories
when it is absent.

The compatibility contract checks:

- model id: `tencent/HunyuanOCR`
- runtime ABI major: `0`
- runtime version range: `[0.1.0, 1.0.0)`
- manifest format: `HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1`
- manifest file count: 170
- precision metadata: `fp32`
- JPEG pixel contract: `stb_rgb_v1`

## Release Packaging

CPack is enabled for the installed runtime and CLI. The packages intentionally
contain the library, CLI, exported CMake package, headers, and README only; the
multi-GiB model directory remains an external artifact validated by manifest and
compatibility metadata.

| Platform | Package | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Linux | `HunyuanOCR-ncnn-0.1.0-Linux-x86_64.tar.gz` | 7,415,840 | `ce619721eccd582772798a3de7483489eef55a859843e05831cdf68a26da9049` |
| Linux | `HunyuanOCR-ncnn-0.1.0-Linux-x86_64.zip` | 7,419,327 | `25d765a858bd92808ec1f8481a6b1145e6615944cae07ce1a1df1eb1679fcdf8` |
| Windows | `HunyuanOCR-ncnn-0.1.0-Windows-AMD64.zip` | 4,417,846 | `4f57a0a7cc98713d05f519bf59257f93927fb564e332454ee4f73f9b1c99a40f` |

## Linux Acceptance

`tools/release/validate_release.py` passed on WSL2 Ubuntu 24.04. It refreshed
manifest/compatibility metadata, configured and built the runtime, installed it,
ran all six CTest suites, validated an independent
`find_package(HunyuanOCR)` consumer, generated TGZ and ZIP packages, and wrote
the dependency/license audit.

Persistent evidence:

- Report: `docs/linux_phase4f_release_validation.json`
- Logs: `/home/asus/hunyuanocr-recovery/phase4f/linux`
- Packages: `/home/asus/hunyuanocr-recovery/phase4f/linux-packages`

## Native Windows Acceptance

`tools/windows/validate_phase4f_msvc.ps1` passed with MSVC 19.51. It built and
installed ncnn, verified the ncnn rotary/RMSNorm precision tests, built and
installed `HunyuanOCR::runtime` and `hunyuanocr_cli`, ran all six CTest suites,
validated the independent CMake consumer, generated the ZIP package, and
recorded PE/dll dependencies through `dumpbin`.

The final report generation reused CTest logs from the immediately preceding
full run after a PowerShell report-writing fix. Each reused log contains
`100% tests passed`.

Persistent evidence:

- Report: `docs/windows_phase4f_release_validation.json`
- Logs and packages: `D:\hunyuanocr-recovery\phase4f`

## Dependency And License Audit

The runtime deploy path remains deliberately small:

- C++17 standard library
- ncnn, linked through `find_package(ncnn CONFIG)`
- OpenMP runtime where enabled by ncnn/toolchain
- `third_party/stb/stb_image.h`, detected as public-domain stb_image

Both Linux and Windows audits completed as `passed_with_warnings` because this
repository still has no top-level `LICENSE` or `COPYING` file, and the installed
ncnn package used for validation did not include a nearby license file. These
are release-process warnings, not OCR parity failures.

## Remaining Risks

- Add the repository's intended top-level license before any public binary
  release.
- Bundle or cite the exact ncnn license file with release artifacts.
- The full Windows CTest matrix is slow because real PNG/JPEG OCR runs long
  packed/unpacked CPU inference.
- `runtime_compatibility.tsv` is optional for backward compatibility; a future
  breaking release should decide when to make it mandatory.
