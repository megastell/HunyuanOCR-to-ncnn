# Decoder 24-layer Step 3 Reference Milestone

## Scope

This milestone captures and validates the third single-token decode
iteration for all 24 HunyuanOCR-1.5 decoder layers using PyTorch
CPU FP32 execution.

## Token contract

- prefill output token: 93892
- Step 1 output token: 5112
- Step 2 output token: 206
- Step 3 input token: 206
- Step 3 output token: 1717

The validated sequence prefix is:

93892 -> 5112 -> 206 -> 1717

## Cache contract

- prefill length: 313
- Step 1 present length: 314
- Step 2 present / Step 3 past length:
  315
- Step 3 present length: 316
- maximum Key history-prefix error:
  0.0000000000e+00
- maximum Value history-prefix error:
  0.0000000000e+00

## Hidden-state contract

The Layer 0 hidden state captured for Step 3 is exactly the input
Embedding output for token 206.

- Embedding versus Layer 0 hidden maximum error:
  0.0000000000e+00

Every layer output is passed directly to the next layer input.

- maximum inter-layer hidden-boundary error:
  0.0000000000e+00

## Tensor shapes

Each of the 24 layers uses:

- hidden state: `[1, 1, 1024]`
- attention mask length: `316`
- past Key/Value: `[1, 8, 315, 128]`
- present Key/Value: `[1, 8, 316, 128]`

## Validation result

All 24 layers satisfy:

- unchanged Key history prefix
- unchanged Value history prefix
- exact Layer-to-Layer hidden handoff
- exact token-206 Embedding contract
- Step 3 output token 1717

## Next milestone

The next milestone must export and validate fixed-shape Step 3
TorchScript, PNNX, and ncnn decoder-layer models using:

- past cache length: 315
- attention mask length: 316
- present cache length: 316
- expected output token: 1717
