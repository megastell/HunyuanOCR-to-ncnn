# Decoder 24-layer Two-step ncnn Chain Milestone

## Scope

This milestone validates two consecutive single-token decode
iterations across all 24 HunyuanOCR-1.5 decoder layers.

## Runtime topology

Step 1 hidden chain:

- Layer 0 output is cloned into Layer 1 input
- this continues through Layer 23
- intermediate hidden states are not reloaded from reference files

Cross-step KV handoff:

- Step 1 Layer i present key is cloned into Step 2 Layer i past key
- Step 1 Layer i present value is cloned into Step 2 Layer i past value
- KV is never transferred between different layer indices
- Step 2 does not reload inference KV from reference files

Step 2 hidden chain:

- Layer 0 output is cloned into Layer 1 input
- this continues through Layer 23
- intermediate hidden states are not reloaded from reference files

## Runtime

- CPU FP32
- threads: 9
- packing layout: enabled
- Vulkan: disabled
- ncnn version: 20260802

## Cache lengths

- Step 1: 313 -> 314
- Step 2: 314 -> 315

## Maximum errors

### Step 1

- input boundary: 2.8312206268e-07
- layer output: 7.1525573730e-07
- present key: 3.9935112000e-06
- present value: 2.3841857910e-07

### Cross-step KV handoff

- key: 3.9935112000e-06
- value: 2.3841857910e-07

### Step 2

- input boundary: 4.7683715820e-07
- layer output: 4.7683715820e-07
- present key: 7.3909759521e-06
- present value: 2.3841857910e-07
- appended key: 7.3909759521e-06
- appended value: 2.2351741791e-07

## Cache integrity

- maximum Step 2 key-prefix error:
  0.0
- maximum Step 2 value-prefix error:
  0.0

The historical 314-token cache prefix is preserved exactly in
every Step 2 layer.

## Result

The complete 24-layer decoder body has been validated across two
consecutive decode iterations with direct hidden-state and
same-layer KV-cache handoff.

This does not yet constitute complete autoregressive generation.

The initial hidden state for each decode iteration is still loaded
from a PyTorch reference tensor. Token embedding feedback has not
yet been integrated.

Final RMSNorm and LM Head have also not yet been attached to the
Step 2 Layer 23 output in this two-step runtime.
