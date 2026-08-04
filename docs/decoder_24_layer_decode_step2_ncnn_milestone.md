# Decoder 24-layer Decode Step 2 ncnn Milestone

## Scope

This milestone validates all 24 fixed-length HunyuanOCR-1.5
Decoder Step 2 graphs independently with ncnn.

Each graph uses:

- current-token length: 1
- past-KV length: 314
- present-KV length: 315
- KV heads: 8
- head dimension: 128

## Runtime

- CPU FP32
- threads: 9
- packing layout: enabled
- Vulkan: disabled
- ncnn version: 20260802

## Export validation

All 24 layers passed:

- PyTorch eager wrapper parity
- TorchScript trace parity
- PNNX conversion
- fixed-length ncnn graph audit
- packed ncnn C++ numerical parity

Eager and TorchScript errors were exactly zero for every layer.

Both attention reshape operations in every Step 2 ncnn graph
are fixed to sequence length 315.

## ncnn maximum errors across all 24 layers

- layer output: 2.3841857910e-07
- present key: 2.8610229492e-06
- present value: 7.4505805969e-08
- appended key: 2.8610229492e-06
- appended value: 7.4505805969e-08

## Cache integrity

Across all 24 layers:

- maximum key-history-prefix error:
  0.0
- maximum value-history-prefix error:
  0.0

The first 314 positions of every ncnn present-KV output are
therefore exact copies of the supplied past-KV input.

## Result

All 24 fixed-length Decoder Step 2 ncnn graphs are independently
validated for packed FP32 CPU execution.

This milestone does not yet validate the complete two-step
24-layer runtime chain. During this milestone, each layer still
receives its own PyTorch reference hidden state and reference
past KV.

The next milestone must directly chain:

- Step 1 layer output to the next Step 1 layer input
- Step 2 layer output to the next Step 2 layer input
- each layer's Step 1 present KV to the same layer's Step 2 past KV
