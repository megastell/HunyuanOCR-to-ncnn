# C++ Prompt and Multimodal Position Milestone

## Scope

Phase 3C removes captured prompt and positional tensors from the fixed OCR
smoke inference path. C++ now performs the following work before ncnn prefill:

- construct the fixed HunyuanOCR user chat turn and special tokens;
- tokenize the Chinese OCR instruction with ByteLevel BPE;
- expand the image placeholder from `image_grid_thw`;
- generate the 313-element attention and multimodal token-type masks;
- generate four-axis position IDs and dynamic mRoPE cosine/sine tensors;
- generate each decode step's causal mask and mRoPE tensors through EOS.

## Audited Contract

- Fixed prompt: `请逐行识别图片中的所有文字。只输出图片中的文字本身，保留换行，不要解释。`
- Tokenizer: BPE with 120000 base vocabulary entries and 119758 merges.
- Pretokenizer: numeric split, CJK split, general Unicode split, then
  ByteLevel encoding without a prefix space.
- Chat-template prefix: BOS `120000`, image-start `120118`.
- Image token: `120120`, repeated 288 times at `[2, 290)`.
- Chat-template suffix: image-end `120119`, 21 BPE prompt tokens, and user
  marker `120006`.
- Sequence length: 313; attention mask is all one; multimodal type is one only
  for `[2, 290)`.
- mRoPE axes: ordinary sequence position, image width, image height, and image
  index; section sizes are `[16, 16, 16, 16]`.
- Dynamic RoPE parameters: theta `10000`, alpha `1000`, head dimension `128`.

## Implementation

- `tools/export/export_tokenizer_vocab.py` now exports both the ID-ordered
  ByteLevel vocabulary and BPE merge ranks into ignored model artifacts.
- `tools/reference/capture_cpp_prompt_mrope_contract.py` reconstructs the
  prompt and positional contract in the pinned Transformers environment and
  compares every field against the established Phase 2/3B captures.
- `tests/multimodal_full_generation/cpp/prompt_inputs.cpp` implements the
  fixed-prompt CJK pretokenization, ByteLevel byte mapping, ranked BPE merge,
  chat-template assembly, masks, four-axis positions, and dynamic mRoPE.
- `prompt_inputs_contract` is a separate audit executable. The main inference
  executable does not load captured prompt, mask, position, or RoPE inputs.
- `tools/validate/validate_cpp_prompt_mrope.sh` performs reference recapture,
  a persistent clean build, contract comparison, both runtime modes, and the
  machine-readable validation report.

## Contract Result

- `input_ids`, attention mask, `mm_token_type_ids`, position IDs, and causal
  mask are exact matches.
- Image token span is exactly `[2, 290)` with 288 image tokens.
- Prefill mRoPE cosine and sine maximum absolute error: `5.96046e-08`.
- Decode Steps 1 through 10 masks are exact; mRoPE maximum absolute error is
  `5.96046e-08`.
- Runtime source audit finds no captured prompt or positional tensor load in
  `main.cpp` or `multimodal_input.cpp`.

## Full Generation Result

Packed and unpacked modes both produce first token `93892`, then the exact
reference sequence:

```text
93892 5112 206 1717 21 185 18009 15613 16678 21836 120007
```

Both modes reach EOS and decode exactly to:

```text
HELLO 2026
NCNN CPU TEST
```

Packed prefill/logits maximum errors remain within the established Phase 3B
tolerances (`3.906e-1` hidden, `3.549e-1` cache, `8.081e-3` logits). Unpacked
errors are `1.120e-3`, `1.459e-3`, and `5.054e-5`, respectively. Every decode
step preserves the existing KV prefix byte for byte.

## Persistent Evidence

- Reference contract: `docs/cpp_prompt_mrope_reference.json`.
- Validation report: `docs/cpp_prompt_mrope_validation.json`.
- Tokenizer export report: `docs/tokenizer_vocab_export.json`.
- Validation logs: `~/hunyuanocr-recovery/phase3c/`.
- Reproduction entry point: `tools/validate/validate_cpp_prompt_mrope.sh`.

## Remaining Boundary

The tokenizer intentionally supports the fixed OCR prompt required by this
milestone. It uses the complete model vocabulary and merge table, but its CJK
pretokenizer rejects other prompt strings rather than claiming general Unicode
chat coverage. The next stable phase should productize the proven components
into a root CMake runtime library and OCR CLI, then broaden prompt/tokenizer
coverage as part of that public API. Native Windows/MSVC build and execution
also remain unverified.
