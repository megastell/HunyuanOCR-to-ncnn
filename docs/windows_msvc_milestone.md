# Native Windows MSVC Runtime Milestone

## Scope

Phase 4B validates the installed HunyuanOCR C++ runtime and OCR CLI on native
Windows x64. The same source tree, converted model files, runtime manifest,
smoke image, token contract, and ncnn precision fixes are used on Linux and
Windows. No captured PyTorch tensor is loaded by the product path.

## Audited Environment

- Windows 11 Pro `10.0.26200`, x64.
- Visual Studio Community 2026 `18.6.2`.
- MSVC C/C++ compiler `19.51.36246`.
- CMake `4.4.0` and Visual Studio Ninja `1.13.2`.
- 31.72 GiB physical RAM and a 25.56 GiB page file.
- ncnn `20260806`, CPU-only, static, OpenMP enabled.
- HunyuanOCR and ncnn source trees were read through
  `\\wsl.localhost\Ubuntu-24.04`; native build and install trees were stored
  under `D:\hunyuanocr-recovery\phase4b`.

## Required ncnn Changes

The native build used the same clean ncnn branch as Linux at commit
`65258a1e0c1b68d8bae57332f883fb3c429e5192`, based on upstream
`a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a` plus:

- `6cc4ef9d`: full-width cosine/sine support for 2D vision RotaryEmbed. The
  x86 implementation uses the same guarded cache-width contract as the
  portable implementation and contains no compiler-specific extension.
- `65258a1e`: precise packed x86 RMSNorm reciprocal square root. The change
  uses MSVC-supported SSE/AVX intrinsics (`sqrt` followed by division).

MSVC compiled all generated x86 dispatch variants, including AVX, AVX2, and
AVX512. Native `test_rotaryembed.exe` and `test_rmsnorm.exe` both returned
exit code zero.

## Build and Package Contract

The root project builds and installs:

- `hunyuanocr_runtime.lib` and the exported `HunyuanOCR::runtime` target.
- `hunyuanocr_cli.exe`.
- Public headers and `HunyuanOCRConfig.cmake` package files.

An independent native consumer configured with `find_package(HunyuanOCR
CONFIG REQUIRED)`, linked `HunyuanOCR::runtime`, and ran successfully against
the installed HunyuanOCR and ncnn prefixes.

The CLI now reports native Windows peak working set through
`GetProcessMemoryInfo`. The only Windows-specific link addition is `Psapi`;
Linux retains `getrusage` and its existing behavior.

## Manifest and Paths

The model and image were consumed directly through UNC paths into WSL. This
validated the complete path chain through `std::filesystem`, C++ file streams,
stb image loading, and ncnn parameter/weight loading.

The same `HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1` file contains 162 entries and
6,076,349,856 bytes. Native C++ size verification passed for unpacked mode,
and native C++ SHA-256 verification passed for every entry before the packed
run.

## Exact Output

Native packed and unpacked runs both emitted:

```text
93892 5112 206 1717 21 185 18009 15613 16678 21836 120007
```

Both reached EOS `120007` and decoded exactly to:

```text
HELLO 2026
NCNN CPU TEST
```

## Measurements

All runs used Release builds and 9 CPU threads. Windows phase timings include
on-demand submodel loading from the WSL UNC path, so they are not pure kernel
benchmarks.

| Platform / mode | Verification | Load | Input | Prefill | Decode | Runtime | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Windows unpacked | size | 3.200 s | 23.471 s | 7.926 s | 8.696 s | 40.093 s | 3,869,608 KiB |
| Windows packed | SHA-256 | 246.654 s | 13.737 s | 8.131 s | 13.624 s | 35.492 s | 5,128,616 KiB |
| Linux unpacked | size | 0.422 s | 7.756 s | 1.231 s | 2.150 s | 11.138 s | 4,251,012 KiB |
| Linux packed | size | 0.431 s | 3.506 s | 1.471 s | 7.404 s | 12.381 s | 5,990,504 KiB |

Windows full SHA/load time was dominated by reading 6.08 GB through the WSL
UNC bridge; the comparable Linux SHA/load time was 16.940 seconds. A release
installation with model files on NTFS must be benchmarked before treating the
UNC numbers as native storage performance.

Windows peak working set and Linux `ru_maxrss` are different operating-system
measurements. Both establish the process peak, but small cross-platform
differences should not be interpreted as allocator savings without a dedicated
memory trace.

## Binary Dependencies

`hunyuanocr_cli.exe` is a PE32+ x64 large-address-aware console application.
ncnn and HunyuanOCR are statically linked. `dumpbin /dependents` reports only:

- Windows `KERNEL32.dll` and Universal CRT API sets.
- `MSVCP140.dll`, `VCRUNTIME140.dll`, and `VCRUNTIME140_1.dll`.
- `VCOMP140.DLL` for MSVC OpenMP.

There is no dynamic ncnn, image codec, tokenizer, or hashing dependency. A
redistributable package must include or require the matching MSVC and OpenMP
runtimes.

## Reproduction and Evidence

Run the native build and validation from Windows PowerShell without changing
the machine execution policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  tools\windows\build_msvc.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  tools\windows\validate_msvc.ps1
```

- Machine-readable report: `docs/windows_msvc_validation.json`.
- Build script: `tools/windows/build_msvc.ps1`.
- Validation script: `tools/windows/validate_msvc.ps1`.
- Native configure, build, test, consumer, dependency, and OCR logs:
  `D:\hunyuanocr-recovery\phase4b`.
- Linux regression logs: `~/hunyuanocr-recovery/phase4a`.

## Remaining Boundary

The two-platform fixed smoke contract is complete. Before public release, the
runtime still needs arbitrary supported image-grid handling and a broader
exact-text regression set. Windows performance should be remeasured with a
manifest-selected model copy on native NTFS, and the two ncnn precision commits
must be carried reproducibly or resolved upstream.
