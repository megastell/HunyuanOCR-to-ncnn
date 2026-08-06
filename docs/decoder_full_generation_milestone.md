# Dynamic Full-Generation Milestone

Date: 2026-08-06

## Result

HunyuanOCR-ncnn now completes the full greedy decoder sequence from the pinned
PyTorch prefill state with one reusable C++ loop and one dynamic ncnn model per
decoder layer.

Generated token sequence:

`93892 5112 206 1717 21 185 18009 15613 16678 21836 120007`

Generated text:

```text
HELLO 2026
NCNN CPU TEST
```

Both packed and unpacked layouts pass. EOS token 120007 is produced at Step 10
and terminates generation.

## Implementation

- `capture_decoder_autoregressive_steps.py` captures Steps 4 through 10 in one
  PyTorch model session. It persists every layer boundary, attention mask, RoPE
  input, past/present KV cache, logits, and token contract.
- The 24 dynamic decoder models are loaded once and reused for all ten decode
  steps. No Step 4 through Step 10 decoder weights are exported.
- The C++ loop owns the live per-layer KV caches and passes each present cache
  directly into the next step.
- Final RMSNorm, LM Head, greedy argmax, token feedback through ncnn Embed, EOS
  handling, and token accumulation all execute in C++.
- `export_tokenizer_vocab.py` exports the tokenizer vocabulary as ID-ordered,
  hex-encoded UTF-8 tokens. The C++ runtime implements the GPT-2 ByteLevel
  reverse byte map and has no tokenizer library or JSON dependency.

## Verification

The Step 4 through Step 10 PyTorch capture contains 168 layer reports. Hidden
boundaries, embedding handoffs, KV history prefixes, and independently
recomputed tail logits all have zero reference-generation error.

For the packed ncnn run, the largest Step 10 hidden error is `1.431e-06`, the
largest cache error across the run is `7.629e-06`, and every live KV prefix is
byte-identical. For the unpacked run, the corresponding values are `9.537e-07`
and `5.960e-06`. Both runs produce the exact token sequence and text.

## Boundary

This milestone starts from captured PyTorch prefill hidden state and KV caches.
The remaining end-to-end blocker is ncnn prefill, followed by the vision tower,
C++ image preprocessing, and input tokenization/prompt construction.
