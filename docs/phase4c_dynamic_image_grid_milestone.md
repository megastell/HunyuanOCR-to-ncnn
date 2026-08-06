# Phase 4C: Dynamic Image Grids and Dual-Platform Regression

## Status

Phase 4C passed on WSL/Linux and native Windows x64/MSVC. The production CLI
now derives the image grid, image-token span, prompt length, four-axis mRoPE
positions, prefill shapes, and decode offset from each input image.

## Removed Fixed Contracts

The previous runtime and converted graphs assumed all of the following:

- image grid `[1,22,50]`;
- 1,100 vision patches;
- 288 merged image tokens in span `[2,290)`;
- prefill sequence length 313.

The dynamic contract is now:

```text
patch_count       = grid_h * grid_w
merged_h          = grid_h / 2
merged_w          = grid_w / 2
image_token_count = merged_h * (merged_w + 1) + 2
image_token_span  = [2, 2 + image_token_count)
prefill_length    = image_token_count + 25
```

The `+1` merged column is the learned newline vector. The final `+2` tokens
are the learned image begin and image end vectors.

## Implementation

- C++ smart resize and patch packing allocate from the actual resized grid.
- The patch projection accepts a dynamic patch count.
- C++ bilinearly interpolates the exported 128x128 learned vision position
  table with `align_corners=false` semantics.
- All 27 vision blocks use dynamic attention and reshape dimensions.
- The patch merger is split into dynamic pre-RMS, convolution, projection,
  and post-RMS ncnn components; C++ inserts newline and boundary vectors.
- The prompt builder dynamically creates image placeholders, token types,
  four-axis position IDs, causal masks, and mRoPE cosine/sine tensors.
- All 24 prefill graphs use dynamic sequence dimensions and return dynamic KV
  caches. Decode starts at the actual prefill length.
- The runtime manifest contains 170 selected files totaling 6,146,592,586
  bytes, including the learned position table and merger constants.

## Exact References

| Case | Grid | Image span | Prefill | Exact text |
|---|---:|---:|---:|---|
| wide | `[1,16,64]` | `[2,268)` | 291 | `WIDE OCR\nTEST 314` |
| square | `[1,32,32]` | `[2,276)` | 299 | `SQUARE\nOCR TEST\n7` |
| tall | `[1,48,24]` | `[2,316)` | 339 | `TALL\nOCR\nTEST 42` |
| smoke | `[1,22,50]` | `[2,290)` | 313 | `HELLO 2026\nNCNN CPU TEST` |

The three new references were captured from the pinned PyTorch model revision
`449e7d471a8a1ef5bd5d652e4881183d7252cbc7`. Saved reference tensors include
processor inputs, vision embeddings, fused embeddings, and position IDs.

## Validation

Linux passed all three dynamic cases in packed and unpacked modes with exact
grid, span, prefill length, token IDs, EOS, and text. Runtime was 10.902 to
20.073 seconds and peak RSS was 3,699,188 to 5,806,220 KiB. The original smoke
case also passed unpacked size verification and packed SHA-256 verification.

Native Windows 11 Pro with MSVC 19.51 passed:

- ncnn `test_rotaryembed` and `test_rmsnorm`;
- runtime build and install;
- an independent `find_package(HunyuanOCR)` consumer;
- all four image cases in packed and unpacked modes;
- C++ SHA-256 manifest verification on the packed smoke run.

Windows dynamic-case runtime was 12.177 to 20.343 seconds and peak RSS was
3,419,756 to 5,338,056 KiB. The packed smoke SHA-256 run loaded in 23.251
seconds, inferred in 13.344 seconds, and peaked at 5,127,892 KiB.

The 170 manifest-selected files were copied from WSL to
`D:\hunyuanocr-recovery\phase4c\model-ntfs`. Copy time was 9.582 seconds and
independent SHA-256 verification of the NTFS copy took 11.277 seconds.

## Persistent Evidence

- PyTorch references: `reference/dynamic_image_cpu_fp32/`
- exact expected outputs: `tests/assets/dynamic_ocr_expected.json`
- reference report: `docs/dynamic_image_reference.json`
- Windows report: `docs/windows_phase4c_validation.json`
- Linux logs: `~/hunyuanocr-recovery/phase4c/`
- Windows build, inference, and copy logs:
  `D:\hunyuanocr-recovery\phase4c\`

## Remaining Risks

- Processor-valid grids near the theoretical maximum remain expensive because
  vision self-attention scales quadratically with patch count. The shape
  contract is dynamic, but practical maximum size depends on available RAM.
- The production decoder currently supports PNG input through stb_image.
- Packed execution is faster in several tested cases but has higher peak RSS.
- The 128x128 vision position table is loaded for each recognition call; caching
  it in the runtime instance would reduce repeated I/O and allocations.
