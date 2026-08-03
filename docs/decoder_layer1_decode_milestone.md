# Decoder Layer 1 Decode FP32 Milestone

## Scope

This milestone validates the true single-token decode path of
HunyuanOCR-1.5 Decoder Layer 1 through:

PyTorch reference → TorchScript → PNNX → ncnn → C++ CPU

## Runtime

- Device: CPU
- Precision: FP32
- Threads: 9
- Vulkan: disabled
- ncnn version: 20260802
- ncnn prefix: ~/.local/ncnn-cpu-ropefix-rmsnorm
- use_packing_layout: true

## Decode contract

Inputs:

- hidden_states: [1, 1, 1024]
- attention_mask: [1, 1, 1, 314]
- rope_cos: [4, 1, 1, 128]
- rope_sin: [4, 1, 1, 128]
- past_key: [1, 8, 313, 128]
- past_value: [1, 8, 313, 128]

Outputs:

- layer_output: [1, 1, 1024]
- present_key: [1, 8, 314, 128]
- present_value: [1, 8, 314, 128]

## PNNX parity

All three PNNX outputs are element-wise identical to the
PyTorch reference:

- layer_output maximum absolute error: 0
- present_key maximum absolute error: 0
- present_value maximum absolute error: 0

## ncnn C++ packed parity

- layer_output maximum absolute error: 5.9604644775e-08
- layer_output mean absolute error: 6.4894720708e-09

- present_key maximum absolute error: 1.1920928955e-06
- present_key mean absolute error: 6.2002864272e-10

- present_value maximum absolute error: 2.9802322388e-08
- present_value mean absolute error: 1.5227881235e-11

Cache checks:

- key history prefix maximum error: 0
- value history prefix maximum error: 0
- key appended token maximum error: 1.1920928955e-06
- value appended token maximum error: 2.9802322388e-08

## Determinism

The packed ncnn C++ parity program was executed nine times.

All nine runs:

- exited successfully
- used nine CPU threads
- used packing layout
- produced identical numerical summaries

## Result

Decoder Layer 1 true decode is validated successfully across
PyTorch, TorchScript, PNNX, and packed ncnn C++ CPU execution.
