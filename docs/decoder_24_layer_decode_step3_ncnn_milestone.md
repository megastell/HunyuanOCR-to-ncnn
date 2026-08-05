# Decoder 24-layer Decode Step 3 ncnn Milestone

## Scope

This milestone validates the third fixed-shape single-token
decode graph for all 24 HunyuanOCR-1.5 decoder layers.

## Token contract

- Step 3 input token: 206
- Step 3 output token: 1717

## Shape contract

- current-token length: 1
- attention-mask length: 316
- past-KV length: 315
- present-KV length: 316
- KV heads: 8
- head dimension: 128

## Export pipeline

All 24 layers passed:

- PyTorch eager reference parity
- TorchScript trace parity
- PNNX conversion
- fixed-shape graph audit
- packed ncnn C++ parity

Layer 0 additionally passed unpacked ncnn execution.

## Maximum ncnn errors

- layer output:
  7.1525573730e-07
- present Key:
  2.8610229492e-06
- present Value:
  5.9604644775e-08
- Key history prefix:
  0.0000000000e+00
- Key appended token:
  2.8610229492e-06
- Value history prefix:
  0.0000000000e+00
- Value appended token:
  5.9604644775e-08

## Result

All same-layer 315-token KV histories remain unchanged while
the 316th token is appended.

This milestone validates fixed-shape Step 3 execution. It does
not yet constitute a dynamic-length generation loop.
