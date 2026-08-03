# Decoder Layer 0 to Layer 23 ncnn Chain Milestone

## Scope

This milestone validates the complete HunyuanOCR-1.5
24-layer decoder during single-token decode using separate
ncnn FP32 layer networks.

The actual execution path is:

Layer 0 out0
→ Layer 1 in0
→ Layer 2 in0
→ ...
→ Layer 23 in0
→ Layer 23 out0

## Direct tensor handoff

Intermediate decoder hidden states are not reloaded from
PyTorch, NumPy, or binary reference files.

The runtime configuration reports:

- intermediate reload: disabled
- handoff: previous out0.clone() -> next in0

Each layer output is cloned into an independently owned
ncnn::Mat before the current layer network is released.

That ncnn::Mat becomes the next layer's real inference input.

Reference hidden-state tensors are loaded only for numerical
validation.

## Runtime

- Device: CPU
- Precision: FP32
- Decoder layers: 24
- Layer range: 0 to 23
- Threads: 9
- Vulkan: disabled
- use_packing_layout: true
- ncnn version: 20260802
- ncnn prefix: ~/.local/ncnn-cpu-ropefix-rmsnorm
- compiler warnings: 0

## Numerical results

Maximum accumulated errors across the complete chain:

- maximum input-boundary error: 2.8312206268e-07
- maximum layer-output error: 7.1525573730e-07
- maximum present-key error: 3.9935112000e-06
- maximum present-value error: 2.3841857910e-07

The largest final layer-output error occurs at Layer 23.

The error remains substantially below the configured FP32
acceptance threshold and does not show unstable amplification.

## KV cache integrity

For every decoder layer from 0 through 23:

- key history-prefix maximum error: 0
- value history-prefix maximum error: 0

This confirms that each layer preserves its 313-token cached
history exactly while appending the new decode token.

## Reference-chain audit

The following tensors were audited across all 24 layers:

- adjacent layer hidden-state boundaries
- attention mask
- RoPE cosine tensor
- RoPE sine tensor
- key-cache history prefix
- value-cache history prefix

All audited reference boundaries are element-wise identical.

## ncnn graph audit

All 24 decoder-layer ncnn graphs use:

- magic: 7767517
- layer/blob graph header: 72 87

## Determinism

The complete packed 24-layer ncnn chain was executed nine
times with nine CPU threads.

All nine runs:

- completed successfully
- passed every numerical threshold
- preserved all KV history prefixes exactly
- produced identical per-layer numerical summaries

## Result

The complete HunyuanOCR-1.5 Decoder Layer 0 to Layer 23
single-token decode path is validated as a directly chained
packed ncnn FP32 CPU pipeline.

No intermediate PyTorch, NumPy, or binary hidden-state
conversion is used between decoder layers.
