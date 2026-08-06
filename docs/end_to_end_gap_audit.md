# HunyuanOCR-ncnn End-to-End Gap Audit

Audit date: 2026-08-06

## Definition of done

The project is complete when a CMake-built C++ application accepts the same
image and prompt as the pinned PyTorch reference, runs preprocessing, vision
encoding, language-model prefill, cached autoregressive decoding, and token
decoding with ncnn, and produces exactly the same final text on Linux and
Windows. Python and PyTorch may be used by conversion and validation tools,
but are not runtime dependencies.

## Pinned baseline

- Model: HunyuanOCR-1.5 base AR
- Model revision: `449e7d471a8a1ef5bd5d652e4881183d7252cbc7`
- `model.safetensors` SHA-256:
  `632a1e082c4dd5a3284cf1ffcdba2fdaa06f435762c58c2f34aff0f3bd6c0249`
- Official HunyuanOCR checkout:
  `c55965d3da1e6f41987abec8068f2e70851318bc`
- PyTorch reference: Python 3.12.3, PyTorch 2.12.1 CPU, Transformers
  5.13.0, FP32 eager attention, 9 threads
- Converter: pnnx 20260704, FP32 output
- ncnn baseline: `a4d2ea1d4422c9e849f166fd7a4aefb52f942f6a`
- Required local ncnn runtime commits:
  - `6cc4ef9d`: full-width cos/sin support for 2D and vision RotaryEmbed
  - `65258a1e`: precise reciprocal square root for packed x86 RMSNorm
- Reference image SHA-256:
  `6dfb4ee4e85fe56a06bae1c5669c0eaac8745baf39999d3cd0ec1f6b1ada4266`
- Reference token sequence:
  `93892, 5112, 206, 1717, 21, 185, 18009, 15613, 16678, 21836, 120007`
- Reference text: `HELLO 2026\nNCNN CPU TEST`

## Verified capabilities

| Area | Evidence | Status |
| --- | --- | --- |
| PyTorch golden run | Three deterministic CPU runs, exact expected text | Complete |
| Text embedding | pnnx/ncnn FP32 parity and shared embedding weight checks | Complete |
| Decoder math | All 24 layers validated for the captured decode contracts | Complete |
| Language-model prefill | All 24 layers produce hidden states and initial KV caches in ncnn | Complete |
| Vision tower and patch merger | Patch embedding, all 27 blocks, and the 288-token merger output pass in packed and unpacked ncnn | Complete |
| Dynamic KV length | One model per layer handles past lengths 313 through 322 | Complete |
| KV handoff | Same-layer cache prefix remains byte-identical across steps | Complete |
| Final RMSNorm | pnnx/ncnn FP32 parity | Complete |
| LM head and argmax | Full 120818-logit parity with matching tokens | Complete |
| Three-token chain | Dynamic 24-layer chain returns `5112 -> 206 -> 1717` | Complete |
| Full decoder generation | One C++ loop emits all 11 tokens, reaches EOS, and reproduces the reference text | Complete |
| ByteLevel text decoding | Exported ID-ordered vocabulary and dependency-free C++ decoder reproduce the reference UTF-8 text | Complete |
| Linux build | CMake and GCC build succeeds in Ubuntu 24.04 WSL2 | Partial platform proof |

The dynamic decoder milestone uses 24 pnnx/ncnn models, one per decoder layer.
Every TorchScript model produced zero maximum error against the Step 1, Step 2,
and Step 3 PyTorch tensors. Every ncnn parameter graph contains a dynamic
sequence reshape, and all 24 dynamic ncnn weight files are byte-identical to
their fixed Step 1 counterparts. The complete three-token chain passes with
both packed and unpacked layouts. The full-generation loop extends this proof
through Step 10, reuses the same 24 decoder models for every step, reaches EOS
token 120007, and decodes `HELLO 2026\nNCNN CPU TEST` in both layouts.

## Important boundary

The current executable is not yet image-to-text end to end. The vision chain
starts from captured `pixel_values` and `image_grid_thw`, executes patch
embedding, all 27 vision blocks, and the patch merger in ncnn, and returns the
reference `[1, 288, 1024]` visual embedding. A separate decoder chain starts
from the captured fused Layer 0 prefill hidden state, attention mask, and mRoPE
tensors and completes prefill and autoregressive generation in ncnn. C++ image
preprocessing, prompt tokenization, multimodal embedding placement, and the
direct handoff between these two verified chains remain incomplete.

## Remaining gaps

| Priority | Gap | Acceptance condition |
| --- | --- | --- |
| P0 | Image preprocessing | C++ resize, normalization, patch packing, grid construction, and image-token placement match the processor tensors |
| P0 | Input tokenizer and prompt construction | C++ tokenization, chat-template handling, and special-token insertion match Transformers; output ByteLevel decoding is already complete |
| P0 | Multimodal embedding placement | The 288 ncnn vision embeddings replace the exact image-token span and reproduce the captured `[1, 313, 1024]` prefill hidden state |
| P1 | Product runtime | Non-empty `src/` and `include/`, a root CMake project, a reusable library, and an OCR CLI replace milestone-only test executables |
| P1 | Native Windows | MSVC CMake build and the same smoke image pass outside WSL |
| P1 | Runtime dependency cleanup | Final executable depends on ncnn and the C++ runtime only; image decoding and tokenizer choices are documented |
| P1 | Model packaging | Conversion outputs are deduplicated and packaged without the 30 GiB development reference tree |
| P1 | ncnn compatibility | The two required ncnn changes are rebased, tested, documented, and either carried cleanly or proposed upstream |
| P2 | ncnn_llm integration | A concrete reuse/fork decision is recorded after comparing loader, tokenizer, sampler, and cache ownership patterns |
| P2 | Release and article | Reproduction guide, accuracy table, performance data, repository URL, and GitHub Discussion are ready for review |

## Phased roadmap

### Phase 1: Complete decoder-side text generation (complete)

1. Capture Steps 4 through 10 from the pinned PyTorch run without exporting
   step-specific decoder weights.
2. Replace the fixed three-step C++ structure with a reusable generation loop
   over the 24 dynamic decoder models.
3. Add EOS handling, greedy argmax, token accumulation, and a minimal exact
   decoder for the smoke sequence.
4. Require the full 11-token sequence and final text to match in packed and
   unpacked modes.

Completed on 2026-08-06. All seven new reference steps have zero PyTorch
boundary and KV-prefix error. The packed and unpacked C++ runs both emit the
exact 11-token sequence, stop on EOS 120007, and decode the exact reference
text.

### Phase 2: Complete ncnn prefill

1. Convert a reusable prefill contract for all 24 decoder layers.
2. Feed captured text and vision embeddings through ncnn prefill.
3. Validate first-token logits and every layer's initial KV cache.
4. Connect the prefill output directly to the Phase 1 generation loop.

Completed on 2026-08-06. All 24 prefill layers now execute in ncnn and return
their hidden, key, and value outputs. Packed and unpacked runs both compute the
reference first token `93892`, hand the generated caches directly to the
dynamic decoder, reproduce all 11 tokens, and stop at EOS with the exact
reference text. No PyTorch prefill KV or first-step decode hidden tensor is
used as an inference input.

### Phase 3: Complete the vision and input pipeline

1. Split and convert vision patch embedding, 27 transformer blocks, and patch
   merger with persisted boundary references.
2. Implement processor-equivalent image operations in C++.
3. Implement chat-template and tokenizer behavior needed by HunyuanOCR.
4. Remove all captured tensors from the inference path.

Phase 3A completed on 2026-08-06. The fixed `22x50` smoke-image contract now
runs patch embedding, all 27 vision blocks, and the patch merger in ncnn. Both
packed and unpacked modes produce `[1, 288, 1024]` embeddings within the final
merger tolerance. The implementation also establishes that this vision model
uses interpolated learned patch positions and no vision RoPE. Phase 3B must
replace captured processor outputs and place the generated visual embeddings
into the decoder prefill sequence.

### Phase 4: Productize and validate two platforms

1. Create the root CMake project, runtime library, and CLI.
2. Add model-manifest validation and clear runtime diagnostics.
3. Build and run the smoke test on Ubuntu and native Windows/MSVC.
4. Add more images and exact-text regression cases, then measure memory and
   latency.

### Phase 5: Publish

1. Minimize and document dependencies and model conversion steps.
2. Prepare the repository for public reproduction without development-only
   artifacts.
3. Review and publish the GitHub Discussion with the repository URL.
4. Evaluate a focused ncnn_llm or ncnn pull request for reusable runtime work.

## Next action

Phase 3B is the next highest-priority milestone. Vision, prefill, and decode
are independently complete at their captured boundaries. The next
implementation should reproduce image preprocessing in C++, then combine the
ncnn visual embeddings with text embeddings at the exact image-token positions
and validate the resulting Layer 0 prefill hidden state before joining the two
existing C++ chains.
