# C++ Multimodal Full-Generation Milestone

## Scope

This Phase 3B milestone replaces the captured `pixel_values` and fused Layer 0
hidden input with a C++ path that starts from the smoke-test PNG:

- decode `tests/assets/ocr_smoke_en.png` as RGB;
- reproduce the processor's resize, normalization, and patch packing;
- execute the Phase 3A patch embedding, 27 vision blocks, and patch merger;
- execute text embedding for all 313 prompt token IDs;
- place 288 vision embeddings into image-token positions `[2, 290)`;
- feed the resulting `[1, 313, 1024]` hidden state into the 24-layer prefill;
- continue through dynamic KV-cache generation until EOS.

## Audited Processor Contract

- Input image: RGB `768x320`, SHA-256
  `6dfb4ee4e85fe56a06bae1c5669c0eaac8745baf39999d3cd0ec1f6b1ada4266`.
- Smart-resize factor: `32`; output: `800x352`.
- Rescale factor: `1/255`; CLIP mean and standard deviation are captured in
  `docs/multimodal_prefill_input_reference.json`.
- Patch size: `16`; merge size: `2`; image grid: `[1, 22, 50]`.
- Packed processor tensor: `[1100, 768]`.
- Prompt length: `313`; image token ID: `120120`.
- Image-token span: `[2, 290)`; image token count: `288`.
- The captured fused embedding is byte-identical to the existing decoder
  Layer 0 prefill reference.

## Implementation

- `tools/reference/capture_multimodal_prefill_input.py` captures the complete
  processor, prompt, text embedding, vision embedding, placement, and mRoPE
  boundary contract with the pinned reference environment.
- `tests/multimodal_full_generation/cpp/multimodal_input.cpp` uses vendored
  `stb_image` for PNG decoding, with no runtime image-library dependency.
- The C++ bicubic resize reproduces Pillow 12.3.0's two-pass 8-bit fixed-point
  behavior, including 22-bit coefficients and clipping after each pass.
- The vision components are loaded one at a time to bound peak memory.
- Text embeddings are computed by ncnn and are byte-identical to PyTorch.
- The generated multimodal hidden state is passed directly to the existing
  24-layer prefill and dynamic generation loop.
- `tools/validate/validate_multimodal_full_generation.sh` performs a clean
  CMake build and validates packed and unpacked modes from the original PNG.

## Packed Result

- Original and resized RGB maximum error: `0`.
- `pixel_values` maximum error: `4.768e-7`.
- Vision/fused hidden maximum error: `3.646e-2`.
- Maximum 24-layer prefill hidden/KV error: `3.906e-1 / 3.549e-1`.
- Final prefill hidden/RMSNorm/logits error:
  `3.818e-4 / 8.564e-3 / 8.082e-3`.
- Maximum dynamic-decode logits error: `1.242e-2`.
- KV prefix mutation: `0` at all ten decode steps.
- Wall time: approximately `14 s`; peak resident memory: approximately
  `5.70 GiB`.

## Unpacked Result

- Original and resized RGB maximum error: `0`.
- `pixel_values` maximum error: `4.768e-7`.
- Vision/fused hidden maximum error: `3.414e-4`.
- Maximum 24-layer prefill hidden/KV error: `1.120e-3 / 1.455e-3`.
- Final prefill hidden/RMSNorm/logits error:
  `1.818e-6 / 4.387e-5 / 4.864e-5`.
- Maximum dynamic-decode logits error: `6.771e-5`.
- KV prefix mutation: `0` at all ten decode steps.
- Wall time: approximately `13 s`; peak resident memory: approximately
  `4.07 GiB`.

## Exact Generation Result

Both modes produce the exact reference token sequence:

```text
93892 5112 206 1717 21 185 18009 15613 16678 21836 120007
```

Both modes reach EOS and decode exactly to:

```text
HELLO 2026
NCNN CPU TEST
```

## Persistent Evidence

- Reference contract: `docs/multimodal_prefill_input_reference.json`.
- Validation report: `docs/multimodal_full_generation_validation.json`.
- Reference capture log:
  `~/hunyuanocr-recovery/multimodal_prefill_input_capture.log`.
- Configure/build logs:
  `~/hunyuanocr-recovery/multimodal_full_generation_cmake.log` and
  `~/hunyuanocr-recovery/multimodal_full_generation_build.log`.
- Packed/unpacked logs:
  `~/hunyuanocr-recovery/multimodal_full_generation_packed.log` and
  `~/hunyuanocr-recovery/multimodal_full_generation_unpacked.log`.

## Remaining Boundary

This fixed smoke contract still loads the captured 313 prompt token IDs,
attention mask, and mRoPE cosine/sine tensors. The next phase should implement
the tokenizer/chat template and multimodal position construction in C++ so a
runtime prompt no longer depends on captured processor tensors. Linux CPU is
validated here; the portable CMake target still requires a separate Windows
build-and-run milestone.

The packed path's approximately `5.70 GiB` peak resident memory is the main
runtime risk exposed by this milestone. Productization should reduce decoder
model residency and temporary allocations before treating the test executable
as a deployable OCR runtime.
