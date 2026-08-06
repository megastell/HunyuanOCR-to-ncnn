# Phase 4D Runtime Memory And Production Inputs

## Scope

Phase 4D makes repeated OCR calls practical while preserving the Phase 4C
dynamic-grid and exact-token contracts. The production runtime still consumes
only the image and converted model directory; captured tensors remain confined
to reference generation and parity tooling.

## Implemented

- Cache the 72 MiB vision position table, patch-merger constants, and fixed OCR
  prompt token IDs when `Runtime::load()` succeeds.
- Stream one decoder layer at a time by default instead of retaining all 24
  dynamic decoder networks. `RuntimeOptions::cache_decode_weights` preserves an
  explicit high-memory caching mode.
- Release vision-stage temporary tensors as soon as their consumers finish and
  remove unused full-image FP32 RGB copies from preprocessing.
- Decode PNG and JPEG inputs with stb_image.
- Reject a resized image before vision inference when its patch count exceeds
  `max_vision_patches`. The error includes the computed grid, configured limit,
  and estimated FP32 attention-score allocation.
- Add a cross-platform repeated-runtime benchmark and persistent Linux/Windows
  validation reports.
- Add a real OCR receipt fixture from the upstream HunyuanOCR repository,
  pinned provenance, PyTorch CPU FP32 token reference, and a reproducible
  lossless PNG generated from Pillow's JPEG decode.

## Memory Results

All RSS values below come from ten mixed-size OCR calls in one process. Every
iteration reproduced its expected token sequence and text.

| Platform / layout | Before final RSS | After final RSS | Before peak | After peak |
| --- | ---: | ---: | ---: | ---: |
| Linux unpacked | 5,582,648 KiB | 1,092,356 KiB | 5,841,204 KiB | 1,241,896 KiB |
| Linux packed | 7,569,760 KiB | 1,114,900 KiB | 7,847,116 KiB | 1,168,012 KiB |

This reduces Linux final RSS by 80.4% unpacked and 85.3% packed, and peak RSS
by 78.7% unpacked and 85.1% packed. Native Windows repeated runs stabilized at
about 811 MiB current RSS: warm-state growth was 300 KiB unpacked and 380 KiB
packed, with peaks of 1,115,228 KiB and 1,129,668 KiB respectively. There was
no monotonic growth across the ten-call sequences.

## Exactness And Compatibility

- Linux and native Windows packed/unpacked regressions pass the wide, square,
  tall, original smoke, and real-receipt lossless PNG cases through EOS.
- The smoke case remains exactly 11 tokens and decodes to
  `HELLO 2026\nNCNN CPU TEST`.
- The real receipt uses grid `[1,58,22]`, prefill length 375, and 179 generated
  tokens including EOS. Both platforms match the PyTorch CPU FP32 token list
  and decoded text exactly in packed and unpacked layouts.
- The original real JPEG is accepted and runs through EOS on both platforms.
- A 6,720-patch document is rejected at the default 2,048-patch limit before
  ncnn allocation, with an estimated 2,756 MiB attention-score warning.

JPEG token equality is not claimed. Pillow/libjpeg and stb_image produce small
pixel differences for the same lossy source, and the observed generation first
diverges near token 49. Exact cross-runtime OCR parity therefore uses the
lossless PNG fixture created from the reference decoder's pixels; JPEG coverage
proves production-format acceptance and successful end-to-end execution.

## Performance Tradeoff

Streaming decoder weights removes several GiB of retained state but reloads 24
layer files for every generated token. This is the preferred bounded-memory
default, but long outputs are slower, especially on Windows. The real 179-token
Windows case took 308.691 seconds unpacked and 413.430 seconds packed. A future
bounded layer cache or memory-budget policy can trade memory for throughput
without changing numerical behavior.

## Persistent Evidence

- `docs/linux_phase4d_validation.json`
- `docs/windows_phase4d_validation.json`
- `docs/real_ocr_reference.json`
- `tests/assets/real_ocr_assets.json`
- `tests/assets/real_ocr_expected.json`
- Linux command logs: `~/hunyuanocr-recovery/phase4d/`
- Windows command logs: `D:\\hunyuanocr-recovery\\phase4d\`

## Remaining Risks

- Exact token parity for arbitrary JPEG decoders is unresolved because decoded
  pixels are not standardized by the JPEG format.
- Vision attention remains quadratic in patch count; the guard prevents an
  accidental oversized allocation but does not implement tiled attention.
- Default decoder streaming favors memory over long-output latency.
- Testing covers Ubuntu 24.04 WSL2 and Windows 11 x64 MSVC; additional Linux
  distributions, compilers, and physical machines remain unverified.
