# Autoregressive Token Feedback Milestone

## Scope

This milestone validates a two-token autoregressive feedback path
for the HunyuanOCR-1.5 decoder using ncnn CPU FP32 execution.

## Runtime path

Step 1:

- Decoder Layer 0 through Layer 23
- Final RMSNorm
- LM Head
- argmax token 5112

Feedback:

- the actual Step 1 token is placed in an int32 ncnn Mat
- ncnn Embed generates the next-token hidden state
- the generated hidden state is passed directly to Step 2 Layer 0
- the Step 2 initial hidden reference tensor is not used as input

Step 2:

- Decoder Layer 0 through Layer 23
- Final RMSNorm
- LM Head
- argmax token 206

## KV-cache path

- Step 1 cache length: 313 to 314
- Step 2 cache length: 314 to 315
- Step 1 Layer i present KV is passed to Step 2 Layer i
- KV is never transferred between different layer indices
- the historical cache prefix remains exactly unchanged

## Validated contracts

- Step 1 expected token: 5112
- Step 1 actual token: 5112
- Token Embedding input type: int32
- Token Embedding output versus reference: byte-identical
- Step 2 initial hidden reference used as input: false
- Step 2 expected token: 206
- Step 2 actual token: 206
- packed execution: validated
- unpacked execution: validated

## Determinism

Nine packed FP32 executions are compared byte for byte. Their
SHA256 values are recorded in:

`docs/autoregressive_token_feedback_determinism.txt`

## Result

The decoder now contains a real two-token autoregressive feedback
connection:

Step 1 token 5112
-> ncnn Embed
-> Step 2 hidden
-> Step 2 token 206

## Remaining scope

This is not complete HunyuanOCR inference.

The runtime still begins from captured Step 1 hidden state, initial
KV cache, attention masks, and RoPE tensors. Vision encoding,
multimodal prefill, tokenizer integration, dynamic mask/RoPE
generation, and an arbitrary-length generation loop are not yet
implemented.
