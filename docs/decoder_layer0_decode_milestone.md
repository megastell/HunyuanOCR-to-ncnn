# Decoder Layer 0 Decode FP32 Milestone

## Scope

This milestone validates the single-token decode path of HunyuanOCR
Decoder Layer 0 through:

PyTorch reference -> TorchScript -> PNNX -> ncnn -> C++ runtime

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

## ncnn C++ result

Configuration:

- FP32
- CPU
- 9 threads
- use_packing_layout=true
- original generated ncnn param and bin

Maximum absolute errors:

- layer_output: 6.7055225372e-08
- present_key: 2.3841857910e-06
- present_value: 2.9802322388e-08

Cache checks:

- key history prefix maximum error: 0
- value history prefix maximum error: 0
- key appended token maximum error: 2.3841857910e-06
- value appended token maximum error: 2.9802322388e-08

All three outputs passed the numerical parity thresholds.

## Determinism

The packing-enabled C++ parity test was repeated nine times.

All runs:

- returned status 0
- produced identical numerical metrics
- produced the same SHA-256 log digest

Digest:

6675acb09f9061c0549a13cd739b8cdea9f0c931dda52aff530987312b9f122c

## Root causes fixed

### Input rank

The C++ runtime originally supplied:

- hidden_states as dims=1
- attention_mask as dims=1

Correct ncnn layouts are:

- hidden_states: dims=2, w=1024, h=1
- attention_mask: dims=3, w=314, h=1, c=1

The incorrect rank caused packed Gemm to interpret one token as eight rows.

### Full-width RotaryEmbed

The local ncnn runtime includes:

6cc4ef9d RotaryEmbed: support full-width cos/sin cache for 2D / vision RoPE

### Packed RMSNorm precision

The packed x86 RMSNorm path used approximate hardware reciprocal square
root instructions. This introduced approximately 1e-3 error in the
Q/K head RMSNorm outputs.

The local ncnn runtime includes:

65258a1e RMSNorm: improve packed x86 reciprocal sqrt precision

After the fix, present_key maximum error decreased from approximately
8.65e-4 to 2.38e-6.

## Required ncnn runtime

Branch:

experiment/rmsnorm-packed-precise-rsqrt

Commits:

- 6cc4ef9d RotaryEmbed full-width cache support
- 65258a1e precise packed x86 RMSNorm reciprocal square root

Local installation:

~/.local/ncnn-cpu-ropefix-rmsnorm
