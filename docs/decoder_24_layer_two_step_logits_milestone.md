# Decoder Two-step Logits Chain Milestone

## Scope

This milestone validates two consecutive single-token decoder
iterations across all 24 HunyuanOCR-1.5 decoder layers, followed
by Final RMSNorm and LM Head execution for Step 2.

## Runtime topology

Step 1:

- 24 decoder layers execute as a direct hidden-state chain
- cache length changes from 313 to 314
- every layer retains its own present key and value

Cross-step handoff:

- Step 1 Layer i present key is passed directly to
  Step 2 Layer i past key
- Step 1 Layer i present value is passed directly to
  Step 2 Layer i past value
- KV is not transferred between different decoder layers
- Step 2 does not reload inference KV from reference files

Step 2:

- 24 decoder layers execute as a direct hidden-state chain
- cache length changes from 314 to 315
- Layer 23 output is passed directly to Final RMSNorm
- Final RMSNorm output is cloned and passed directly to LM Head
- the tail input is not replaced by a reference hidden tensor

## Runtime

- CPU FP32
- threads: 9
- Vulkan: disabled
- vocabulary size: 120818
- packed and unpacked execution both validated

## Packed maximum errors

- Step 2 Layer 23 boundary:
  4.0233135223e-07
- Final RMSNorm:
  1.2397766113e-05
- LM Head logits:
  1.0490417480e-05
- Step 2 key-history prefix:
  0.0
- Step 2 value-history prefix:
  0.0

## Unpacked maximum errors

- Step 2 Layer 23 boundary:
  3.5762786865e-07
- Final RMSNorm:
  1.0013580322e-05
- LM Head logits:
  9.5367431641e-06
- Step 2 key-history prefix:
  0.0
- Step 2 value-history prefix:
  0.0

## Token contract

- expected Step 2 token: 206
- packed actual Step 2 token: 206
- unpacked actual Step 2 token: 206
- contract token: 206

## Determinism

Nine packed FP32 executions completed successfully. Their complete
text outputs were compared byte for byte and recorded with SHA256
hashes in:

`docs/decoder_24_layer_two_step_logits_determinism.txt`

## Result

The following ncnn CPU FP32 path is validated:

Step 1 Decoder Layer 0-23
-> same-layer KV handoff
-> Step 2 Decoder Layer 0-23
-> Final RMSNorm
-> LM Head
-> 120818-dimensional logits
-> token 206

This is not yet complete autoregressive generation.

The Step 2 initial hidden state is still loaded from its PyTorch
reference tensor. Token embedding feedback for Step 1 token 5112
has not yet been integrated.
