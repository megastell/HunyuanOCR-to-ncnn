# 24-Layer ncnn Prefill Milestone

## Scope

This milestone executes the complete 24-layer HunyuanOCR language-model
prefill in ncnn from the captured fused text/image embedding state. Every
prefill layer has four FP32 inputs and three FP32 outputs:

- hidden state: `[1, 313, 1024]`
- attention mask: `[1, 1, 313, 313]`
- mRoPE cosine and sine: `[4, 1, 313, 128]` each
- output hidden state: `[1, 313, 1024]`
- output key and value: `[1, 8, 313, 128]` each

`position_ids` is `None` at all 24 decoder-layer calls. The four-component
position information is already represented by the captured mRoPE tensors.

## Implementation

- `tools/reference/capture_decoder_24_layer_prefill.py` captures all layer
  boundaries, initial KV caches, final RMSNorm inputs/outputs, and logits from
  the pinned PyTorch reference.
- `tools/export/export_decoder_prefill_kv.py` exports one static prefill model
  per layer and validates hidden/key/value parity before pnnx conversion.
- `tests/decoder_24_layer_full_generation/cpp/main.cpp` loads each prefill
  model one at a time, retains cloned KV outputs, computes the first token, and
  feeds those caches directly into the existing dynamic decode loop.
- The first decode hidden state is produced by the ncnn token embedding for
  token `93892`; it is no longer loaded from a PyTorch reference tensor.

## Reference Validation

- Captured decoder layers: 24/24.
- Layer-to-layer PyTorch boundary maximum error: `0`.
- Captured prefill KV versus the existing decode Step 1 past KV: `0`.
- Layer 0 hidden, attention mask, and mRoPE versus the earlier contract: `0`.
- Expected and actual PyTorch prefill token: `93892`.
- Prefill eager parity metrics with zero maximum error: 72/72.
- Prefill TorchScript parity metrics with zero maximum error: 72/72.
- Successful pnnx/ncnn conversions: 24/24.

## ncnn Packed Result

- Maximum prefill hidden error: `2.289e-05`.
- Maximum initial KV error: `4.339e-05`.
- Final prefill hidden error: `1.550e-06`.
- Final RMSNorm error: `3.052e-05`.
- Prefill logits error: `3.529e-05`.
- First token: `93892`.
- EOS reached with the exact 11-token sequence and reference text.

## ncnn Unpacked Result

- Maximum prefill hidden error: `4.578e-05`.
- Maximum initial KV error: `3.546e-05`.
- Final prefill hidden error: `1.173e-06`.
- Final RMSNorm error: `2.861e-05`.
- Prefill logits error: `3.672e-05`.
- First token: `93892`.
- EOS reached with the exact 11-token sequence and reference text.

Both modes produce:

```text
HELLO 2026
NCNN CPU TEST
```

## Persistent Evidence

- Reference report: `docs/decoder_24_layer_prefill_reference.json`.
- Per-layer export reports: `docs/decoder_layer*_prefill_kv.json`.
- Per-layer pnnx logs: `docs/decoder_layer*_prefill_kv_pnnx.txt`.
- Capture log: `~/hunyuanocr-recovery/decoder_24_layer_prefill_capture.log`.
- Export log: `~/hunyuanocr-recovery/decoder_24_layer_prefill_kv_export.log`.
- Packed runtime log: `~/hunyuanocr-recovery/full_prefill_generation_packed.log`.
- Unpacked runtime log:
  `~/hunyuanocr-recovery/full_prefill_generation_unpacked.log`.

## Remaining Boundary

The C++ runtime still loads the fused Layer 0 prefill hidden state, attention
mask, and mRoPE tensors captured after PyTorch embedding and vision processing.
The next milestone must produce those inputs from the image, prompt, vision
tower, patch merger, tokenizer, and multimodal placement logic.
