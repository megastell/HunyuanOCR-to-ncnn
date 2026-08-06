# Full ncnn Vision Tower Milestone

## Scope

This Phase 3A milestone executes the complete HunyuanOCR vision tower in ncnn
from the captured processor outputs:

- `pixel_values`: `[1100, 768]` FP32
- `image_grid_thw`: `[[1, 22, 50]]` INT64
- patch embedding output: `[1, 1100, 1152]`
- 27 vision Transformer blocks: `[1100, 1152]` internal ncnn contract
- patch merger output: `[1, 288, 1024]`

The merger output is persisted as a direct input for the next multimodal
embedding-placement milestone.

## Audited Vision Contract

- Patch size: `16`, input channels: `3`, patch count: `1100`.
- Vision hidden size: `1152`, intermediate size: `4304`.
- Attention heads: `16`, head dimension: `72`.
- Attention is full and unmasked for the one-image smoke input.
- The vision tower has no RoPE module. It uses a learned `128x128` patch-grid
  position embedding bilinearly interpolated to `22x50`.
- Spatial merge size: `2`, producing an `11x25` feature grid.
- Eleven newline embeddings plus begin/end embeddings produce 288 output
  tokens: `11 * (25 + 1) + 2`.

## Implementation

- `tools/reference/capture_vision_tower_full.py` captures patch embedding,
  every block input/output, and merger boundaries from the pinned PyTorch
  reference environment.
- `tools/export/export_vision_tower_full.py` exports the patch embedding, 27
  blocks, and merger as 29 FP32 pnnx/ncnn components.
- The exporter validates eager and frozen TorchScript outputs before pnnx.
- pnnx 20260704 drops three physical q/k/v transposes while lowering this
  fixed vision SDPA graph. A structure-checked param normalization inserts
  the three required ncnn `Permute` layers and rejects unexpected graphs.
- `tests/vision_tower_full/cpp/main.cpp` loads one component at a time,
  validates all 29 PyTorch boundaries, and writes the final embedding.
- `tools/validate/validate_vision_tower_full.sh` audits all reports and graph
  layouts, rebuilds with CMake, runs packed and unpacked modes, and creates the
  final machine-readable validation report.

## PyTorch Reference Result

- Captured vision blocks: 27/27.
- Layer-to-layer boundary maximum error: `0`.
- Split-contract last-hidden maximum error: `0`.
- Split-contract merger-output maximum error: `0`.
- Vision rotary-module count: `0`.
- Successful pnnx/ncnn conversions: 29/29.

## ncnn Packed Result

- Patch embedding maximum error: `1.526e-05`.
- Maximum block-output error: `4.614e+02` at layer 26.
- Layer 26 mean error: `1.059e-02`; cosine similarity: greater than
  `0.99999`.
- Final merger maximum error: `3.597e-02`.
- Final merger mean error: `2.953e-04`.
- Final merger cosine similarity: `0.9999993385`.
- Validation elapsed time: `4.37 s`.
- Peak resident memory: approximately `339 MiB`.

Layer 26 contains sparse reference activations up to `13317.6`, which amplify
packed FP32 accumulation differences. The merger's input RMSNorm reduces this
error to the strict final-output tolerance.

## ncnn Unpacked Result

- Patch embedding maximum error: `1.526e-05`.
- Maximum block-output error: `2.084e+00` at layer 26.
- Final merger maximum error: `2.966e-04`.
- Final merger mean error: `1.437e-06`.
- Final merger cosine similarity: `0.99999999999`.
- Validation elapsed time: `7.55 s`.
- Peak resident memory: approximately `402 MiB`.

Both modes produce the required `[1, 288, 1024]` vision embedding and pass the
same final-output thresholds.

## Multimodal Handoff

The C++ chain now has this explicit interface:

- input: normalized and patch-packed `pixel_values[1100][768]`
- input metadata: `image_grid_thw[1][3] = {{1, 22, 50}}`
- output: `vision_embeddings[1][288][1024]`

Development outputs are written to:

- `artifacts/vision_tower_full/output/vision_embeddings_packed_f32.bin`
- `artifacts/vision_tower_full/output/vision_embeddings_unpacked_f32.bin`

The next runtime stage must place these 288 embeddings into the prompt's image
token positions and reproduce the captured `[1, 313, 1024]` decoder prefill
hidden state.

## Persistent Evidence

- Reference report: `docs/vision_tower_full_reference.json`.
- Export reports: `docs/vision_patch_embed.json`,
  `docs/vision_block0.json` through `docs/vision_block26.json`, and
  `docs/vision_patch_merger.json`.
- Final validation report: `docs/vision_tower_full_validation.json`.
- Reference capture log:
  `~/hunyuanocr-recovery/vision_tower_full_capture.log`.
- Export logs: `~/hunyuanocr-recovery/vision_tower_blocks_export_v2.log`,
  `vision_patch_embed_export_v3.log`, and
  `vision_patch_merger_export_v3.log` in the same directory.
- Complete validation log:
  `~/hunyuanocr-recovery/vision_tower_full_validation.log`.
- Packed and unpacked runtime logs:
  `~/hunyuanocr-recovery/vision_tower_full_packed.log` and
  `~/hunyuanocr-recovery/vision_tower_full_unpacked.log`.

## Remaining Boundary

The chain still loads captured `pixel_values` and `image_grid_thw`; it does not
yet preprocess the source image in C++. The next Phase 3 milestone must match
resize, normalization, patch packing, prompt tokenization, and multimodal
embedding placement before connecting this output to the complete ncnn
prefill and generation path.
