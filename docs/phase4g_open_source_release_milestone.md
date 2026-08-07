# Phase 4G Open Source Release Preparation

## Scope

Phase 4G completes the practical open-source release preparation around the
runtime package. It adds a top-level source license, required NOTICE file,
third-party license archive, Linux/Windows installation docs, and clean
extraction rehearsals that run OCR from the packaged CLI.

## Licensing And Notices

The repository source code, runtime library, CLI, tests, CMake packaging, and
project documentation are now licensed under Apache-2.0 through the top-level
`LICENSE` file.

The HunyuanOCR model is not covered by Apache-2.0. The model license is archived
as `third_party/licenses/Tencent-HunyuanOCR-LICENSE.txt`, and the release NOTICE
includes the required Tencent Hunyuan distribution notice. The project also
states that it is not affiliated with, sponsored by, or endorsed by Tencent.

Archived third-party licenses:

| Component | License | Archive |
| --- | --- | --- |
| ncnn | BSD 3-Clause with bundled third-party notices | `third_party/licenses/ncnn-LICENSE.txt` |
| stb_image | Public domain image loader | `third_party/licenses/stb_image-LICENSE.txt` |
| Tencent HunyuanOCR model | Tencent Hunyuan Community License Agreement | `third_party/licenses/Tencent-HunyuanOCR-LICENSE.txt` |

The Phase 4G dependency/license audit passes with zero warnings on both Linux
and Windows.

## Package Contents

CPack now installs:

- `LICENSE`
- `NOTICE`
- `README.md`
- `THIRD_PARTY_NOTICES.md`
- `install_linux.md`
- `install_windows.md`
- `third_party/licenses/*`
- runtime headers
- `HunyuanOCR::runtime` CMake package files
- `hunyuanocr_cli`

The package still intentionally excludes the multi-GiB converted model
directory. The external model directory is validated by `runtime_manifest.tsv`
and `runtime_compatibility.tsv`.

## Package Hashes

| Platform | Package | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Linux | `HunyuanOCR-ncnn-0.1.0-Linux-x86_64.tar.gz` | 7,426,293 | `b13cbfb201c3a0142d78853e1b3748c2738f6cb322caab27cb9af754949e8388` |
| Linux | `HunyuanOCR-ncnn-0.1.0-Linux-x86_64.zip` | 7,439,525 | `0817b4a8fd161f3c99f66d04393dc85b96d8d005a6148e5ceb10c63baa7bff0a` |
| Windows | `HunyuanOCR-ncnn-0.1.0-Windows-AMD64.zip` | 4,437,986 | `31c5a36bbab680bbb8270621d4d1e1e735207018f599f6e817f8ef83cd70ec09` |

## Release Validation

Linux Phase 4G validation:

- built and installed the package
- ran CTest `smoke` and `error-paths`
- validated an independent `find_package(HunyuanOCR)` consumer
- generated TGZ and ZIP packages
- completed dependency/license audit with zero warnings

Windows Phase 4G validation:

- built and installed ncnn and HunyuanOCR-ncnn with MSVC 19.51
- ran ncnn rotary/RMSNorm precision tests
- ran CTest `smoke` and `error-paths`
- validated an independent `find_package(HunyuanOCR)` consumer
- generated the Windows ZIP package
- recorded PE dependencies with `dumpbin`
- completed dependency/license audit with zero warnings

Full OCR matrix coverage for dynamic sizes, real PNG, real JPEG, and cache
budgets remains recorded in the Phase 4F reports. Phase 4G focuses on release
packaging and clean-install acceptance after the documentation and notice
changes.

## Clean Package Rehearsal

Both platforms extracted the final package into a clean directory, verified the
installed release-notice files, copied only the smoke input image into the clean
work directory, and ran OCR from the extracted CLI against the external model
directory.

| Platform | Layout | Decoder Cache | Runtime | Peak RSS | Output |
| --- | --- | ---: | ---: | ---: | --- |
| Linux | unpacked | 512 MiB | 11.867 s | 1,678,716 KiB | `HELLO 2026\nNCNN CPU TEST` |
| Windows | unpacked | 512 MiB | 20.717 s | 1,570,712 KiB | `HELLO 2026\nNCNN CPU TEST` |

## Persistent Evidence

- Linux validation: `docs/linux_phase4g_release_validation.json`
- Windows validation: `docs/windows_phase4g_release_validation.json`
- Linux dry run: `docs/linux_phase4g_release_dryrun.json`
- Windows dry run: `docs/windows_phase4g_release_dryrun.json`
- Linux logs/packages: `/home/asus/hunyuanocr-recovery/phase4g`
- Windows logs/packages: `D:\hunyuanocr-recovery\phase4g`

## Remaining Release Risks

- The public repository still needs a final GitHub-facing cleanup pass before
  publishing: screenshots, release notes, and a concise support statement.
- The model acquisition/conversion pipeline is documented but still depends on
  local generated artifacts; a one-command reproducible conversion pipeline is
  the next high-value improvement.
- The Windows package depends on MSVC/OpenMP runtime DLLs already being
  available on the target machine.
- The binary packages are runtime-only; model artifact hosting and distribution
  policy must be decided separately under the Tencent Hunyuan license.
