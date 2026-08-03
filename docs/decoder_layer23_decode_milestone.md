# Decoder Layer 23 Decode FP32 Milestone

## Scope

This milestone validates the true single-token decode path of
HunyuanOCR-1.5 Decoder Layer 23, the final decoder layer,
through:

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

## Cache contract

- decoder cache layers: 24
- selected layer index: 23
- key history prefix maximum error: 0
- value history prefix maximum error: 0

## TorchScript parity

Eager and TorchScript outputs are element-wise identical:

- layer_output maximum absolute error: 0
- present_key maximum absolute error: 0
- present_value maximum absolute error: 0

## PNNX parity

All three PNNX outputs are element-wise identical to the
PyTorch reference:

- layer_output maximum absolute error: 0
- present_key maximum absolute error: 0
- present_value maximum absolute error: 0

## ncnn C++ packed parity

- layer_output maximum absolute error: 4.7683715820e-07
- layer_output mean absolute error: 3.8247435441e-08
- layer_output RMSE: 5.0702307944e-08

- present_key maximum absolute error: 1.1920928955e-06
- present_key mean absolute error: 6.9168665134e-10
- present_key RMSE: 1.6180236579e-08

- present_value maximum absolute error: 8.1956386566e-08
- present_value mean absolute error: 5.1967194879e-11
- present_value RMSE: 1.1904111485e-09

Cache region checks:

- key history prefix maximum error: 0
- value history prefix maximum error: 0
- key appended token maximum error: 1.1920928955e-06
- value appended token maximum error: 8.1956386566e-08

## Determinism

The packed ncnn C++ parity program was executed nine times.

All nine runs:

- exited successfully
- used nine CPU threads
- used packing layout
- produced identical numerical summaries

## Result

Decoder Layer 23 true decode is validated successfully across
PyTorch, TorchScript, PNNX, and packed ncnn C++ CPU execution.

Together with the validated Layer 0 and Layer 1 paths, this
confirms that the parameterized decode workflow applies to the
first layer, a non-first layer, and the final decoder layer.
