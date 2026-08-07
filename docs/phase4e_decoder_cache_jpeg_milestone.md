# Phase 4E Decoder Cache And JPEG Parity

## Scope

Phase 4E adds a bounded decoder cache that improves long-output latency without
returning to the multi-GiB expanded network cache. It also defines one JPEG RGB
pixel contract for the production runtime and PyTorch reference so lossy-decoder
rounding no longer makes token parity ambiguous.

## Decoder Cache Design

Each of the 24 dynamic decoder layers has a 69,215,260-byte ncnn model file.
The runtime reads a consecutive prefix of these raw files into memory until the
configured `decoder_cache_budget_mib` would be exceeded. A decode step still
constructs only the current non-resident network, but cached layers load through
ncnn's memory reader instead of the filesystem. The sequential prefix avoids the
zero-hit behavior that an LRU cache would have for repeated 0-to-23 layer scans.

The legacy `cache_decode_weights` option remains available and mutually
exclusive with a nonzero byte-cache budget. It retains expanded `ncnn::Net`
objects and is kept for compatibility, not recommended production use.

| Budget | Raw layers cached | Retained decoder bytes |
| ---: | ---: | ---: |
| 0 MiB | 0 | 0 MiB |
| 512 MiB | 7 | 463 MiB |
| 2048 MiB | 24 | 1585 MiB |

The budget governs retained raw decoder bytes only. Total RSS additionally
contains the active decoded layer, text/vision networks, image tensors, KV
caches, allocator state, and platform runtime libraries.

## Linux Results

The real receipt JPEG generates 179 tokens including EOS. Every row below is an
exact PyTorch token and text match.

| Layout | Budget | Runtime | Peak RSS | Change from 0 MiB |
| --- | ---: | ---: | ---: | ---: |
| unpacked | 0 MiB | 89.697 s | 1,188,272 KiB | baseline |
| unpacked | 512 MiB | 65.322 s | 1,660,832 KiB | 27.2% faster |
| unpacked | 2048 MiB | 45.175 s | 2,904,028 KiB | 49.6% faster |
| packed | 0 MiB | 189.308 s | 1,198,848 KiB | baseline |
| packed | 512 MiB | 169.212 s | 1,667,848 KiB | 10.6% faster |
| packed | 2048 MiB | 151.570 s | 2,917,772 KiB | 19.9% faster |

The dynamic wide, square, and tall regressions also pass packed/unpacked with a
512 MiB budget. The original real PNG remains exact for both layouts, and smoke
still decodes to `HELLO 2026\nNCNN CPU TEST` through EOS.

Ten mixed-size calls through one reused runtime also remain stable. Unpacked RSS
ends 48,492 KiB below its post-warmup second iteration; packed grows only 2,780
KiB. Every iteration reproduces its expected token sequence.

## Native Windows Results

MSVC 19.51 builds and installs the runtime, the ncnn precision tests pass, and
the independent consumer still resolves `HunyuanOCR::runtime` through CMake.

| Layout | Budget | Runtime | Peak RSS | Change from 0 MiB |
| --- | ---: | ---: | ---: | ---: |
| unpacked | 0 MiB | 185.925 s | 1,146,172 KiB | baseline |
| unpacked | 2048 MiB | 69.795 s | 2,770,956 KiB | 62.5% faster |
| packed | 0 MiB | 347.517 s | 1,159,752 KiB | baseline |
| packed | 2048 MiB | 174.192 s | 2,782,172 KiB | 49.9% faster |

Windows smoke passes packed/unpacked with the 512 MiB budget. Binary dependency
and build logs remain under `D:\hunyuanocr-recovery\phase4e`.

## JPEG Pixel Contract

Pillow 12.3 uses libjpeg-turbo, while production uses stb_image. Direct decoding
of the official receipt differs in 15,662 of 331,232 pixels, with channel errors
bounded to 2 or 3. That small difference first changes generation at token index
48: the Pillow-derived PNG emits 29977 and the stb JPEG emits 50814.

`hunyuanocr_decode_image_rgb` now calls the same decode function as the runtime
and writes a lossless RGB PPM contract. Its SHA-256 is
`df563d0103382032ce9f4d00f313b3e5a848664fba8dfdfe1f611b8aa84e2de9` on both
Linux and Windows. PyTorch consumes this lossless contract while the production
CLI consumes the original JPEG. Both therefore start from byte-identical RGB,
and packed/unpacked ncnn matches all 179 PyTorch tokens and text on both systems.

This contract deliberately does not claim that Pillow/libjpeg and stb_image
produce identical pixels. It proves model and preprocessing parity for the
production decoder's pixels and preserves the original JPEG as the runtime input.

## Persistent Evidence

- `docs/linux_phase4e_validation.json`
- `docs/windows_phase4e_validation.json`
- `docs/real_ocr_reference.json`
- `tests/assets/real_ocr_assets.json`
- `tests/assets/real_ocr_expected.json`
- Linux logs: `~/hunyuanocr-recovery/phase4e/`
- Windows logs: `D:\hunyuanocr-recovery\phase4e\`

## Remaining Risks

- A 2048 MiB decoder budget raises total process RSS to roughly 2.8 GiB; 512 MiB
  is the more conservative deployment setting.
- Packed layout is slower than unpacked for this FP32 CPU workload on both tested
  systems. It remains a compatibility mode rather than the recommended default.
- The RGB contract is stb_image-specific. Changing the production JPEG decoder
  requires recapturing JPEG references.
- CTest has no registered default tests; end-to-end validation is currently
  driven by the persistent Python and PowerShell parity scripts.
