# Decoder Layer 0 to Layer 1 Decode Chain Milestone

## Scope

This milestone validates direct ncnn tensor handoff between
HunyuanOCR-1.5 Decoder Layer 0 and Decoder Layer 1 during
single-token decode.

The actual execution path is:

Layer 0 ncnn out0 → Layer 1 ncnn in0

Layer 1 does not reload its hidden-state input from a PyTorch,
NumPy, or binary reference tensor.

## Runtime

- Device: CPU
- Precision: FP32
- Threads: 9
- Vulkan: disabled
- ncnn version: 20260802
- ncnn prefix: ~/.local/ncnn-cpu-ropefix-rmsnorm
- use_packing_layout: true
- compiler warnings: 0

## Reference boundary

The captured PyTorch Layer 0 output and Layer 1 hidden-state
input are element-wise identical:

- maximum absolute error: 0
- mean absolute error: 0
- RMSE: 0

The attention mask, RoPE cosine, and RoPE sine reference
tensors are also element-wise identical between both layers.

## Direct ncnn boundary

Actual Layer 0 ncnn out0 compared with the expected Layer 1
hidden-state input:

- maximum absolute error: 6.7055225372e-08
- mean absolute error: 8.7139824245e-09
- RMSE: 1.1400673932e-08
- cosine similarity: 1.0

## Layer 0 parity

- output maximum absolute error: 6.7055225372e-08
- output mean absolute error: 8.7139824245e-09

- present-key maximum absolute error: 2.3841857910e-06
- present-key mean absolute error: 7.6836643205e-10

- present-value maximum absolute error: 2.9802322388e-08
- present-value mean absolute error: 1.4355859742e-11

## Layer 1 chained parity

- output maximum absolute error: 8.1956386566e-08
- output mean absolute error: 1.1458723748e-08
- output RMSE: 1.4889927720e-08

- present-key maximum absolute error: 1.9073486328e-06
- present-key mean absolute error: 1.1121632136e-09

- present-value maximum absolute error: 4.0978193283e-08
- present-value mean absolute error: 2.8040098198e-11

The Layer 0 numerical error did not cause meaningful error
amplification in Layer 1.

## Cache integrity

- Layer 0 key history prefix maximum error: 0
- Layer 0 value history prefix maximum error: 0
- Layer 1 key history prefix maximum error: 0
- Layer 1 value history prefix maximum error: 0

## Determinism

The packed two-layer ncnn C++ chain was executed nine times.

All nine runs:

- exited successfully
- used nine CPU threads
- used packing layout
- passed all numerical thresholds
- produced identical numerical summaries

## Result

Decoder Layer 0 and Decoder Layer 1 are validated as a directly
chained packed ncnn FP32 CPU decode pipeline.

This proves that decoder-layer outputs can be passed directly
between separate ncnn networks without an intermediate
PyTorch, NumPy, or binary tensor conversion.
