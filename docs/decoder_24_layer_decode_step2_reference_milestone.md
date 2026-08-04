# Decoder 24-layer Decode Step 2 Reference Milestone

## Scope

All 24 HunyuanOCR-1.5 decoder layers were captured during one
PyTorch Step 2 decode execution.

The model was loaded once. Step 2 was executed once.

## Token contract

- prefill output token: 93892
- Step 1 output token: 5112
- Step 2 output token: 206

## KV-cache contract

- prefill length: 313
- Step 1 present length: 314
- Step 2 past length: 314
- Step 2 present length: 315
- KV heads: 8
- head dimension: 128

## Validation

- captured decoder layers: 24
- maximum key-prefix error: 0.0
- maximum value-prefix error: 0.0
- model load seconds: 1.314600
- Step 2 decode seconds: 0.053089

All 24 layers satisfy the following reference contracts:

- Step 1 present key equals Step 2 past key byte for byte.
- Step 1 present value equals Step 2 past value byte for byte.
- The historical prefix of each Step 2 present key is unchanged.
- The historical prefix of each Step 2 present value is unchanged.
- Each decoder-layer output equals the next layer hidden input
  byte for byte.
- Attention mask, RoPE tensors, and decode-logits copies are
  consistent across all layer reference directories.

Layer 0 tensors captured by this batch process are byte-identical
to the previously captured independent Layer 0 Step 2 reference.

## Result

The complete 24-layer PyTorch reference contract required for
a two-step ncnn decoder chain is now available.

This milestone validates reference capture only. It does not
yet prove that all 24 fixed-length Step 2 ncnn graphs or the
complete two-step ncnn decoder chain are numerically correct.
