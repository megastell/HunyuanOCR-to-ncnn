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
| Image preprocessing | C++ PNG decode, Pillow-compatible resize, normalization, patch packing, and grid construction | Complete |
| Multimodal embedding placement | 288 vision embeddings replace image-token positions `[2, 290)` and feed ncnn prefill directly | Complete |
| Dynamic KV length | One model per layer handles past lengths 313 through 322 | Complete |
| KV handoff | Same-layer cache prefix remains byte-identical across steps | Complete |
| Final RMSNorm | pnnx/ncnn FP32 parity | Complete |
| LM head and argmax | Full 120818-logit parity with matching tokens | Complete |
| Three-token chain | Dynamic 24-layer chain returns `5112 -> 206 -> 1717` | Complete |
| Full decoder generation | One C++ loop emits all 11 tokens, reaches EOS, and reproduces the reference text | Complete |
| ByteLevel text decoding | Exported ID-ordered vocabulary and dependency-free C++ decoder reproduce the reference UTF-8 text | Complete |
| Fixed OCR prompt encoding | C++ chat assembly and ByteLevel BPE produce all 313 reference input IDs exactly | Complete |
| Multimodal positions | C++ attention/type masks, four-axis position IDs, and prefill/decode mRoPE replace captured positional inputs | Complete |
| Product runtime | Root CMake builds and installs `HunyuanOCR::runtime` plus a reference-free OCR CLI | Complete on Linux |
| Model manifest | 162 product files have size and SHA-256 inventory checks; runtime set is 5.659 GiB | Complete |
| Linux build | CMake/GCC build, install, packed CLI, unpacked CLI, and parity tests pass in Ubuntu 24.04 WSL2 | Complete |

The dynamic decoder milestone uses 24 pnnx/ncnn models, one per decoder layer.
Every TorchScript model produced zero maximum error against the Step 1, Step 2,
and Step 3 PyTorch tensors. Every ncnn parameter graph contains a dynamic
sequence reshape, and all 24 dynamic ncnn weight files are byte-identical to
their fixed Step 1 counterparts. The complete three-token chain passes with
both packed and unpacked layouts. The full-generation loop extends this proof
through Step 10, reuses the same 24 decoder models for every step, reaches EOS
token 120007, and decodes `HELLO 2026\nNCNN CPU TEST` in both layouts.

## Important boundary

The Phase 4A product runtime is image-to-text end to end for the fixed smoke
contract. A root CMake project builds and installs a reusable library and OCR
CLI. The product source contains no captured reference input or output path;
reference tensors are confined to separately enabled parity targets. The
tokenizer still rejects prompts other than the audited OCR instruction, and
the image path currently requires the audited `[1,22,50]` processor grid.

## Remaining gaps

| Priority | Gap | Acceptance condition |
| --- | --- | --- |
| P1 | General prompt API | Extend the audited fixed-prompt tokenizer to the full configured Unicode pretokenizer and multi-turn chat-template surface |
| P1 | Native Windows | MSVC CMake build and the same smoke image pass outside WSL |
| P1 | Memory reduction | Add decoder streaming and shared tied embedding/LM-head weights; measure output parity and latency tradeoffs |
| P1 | Model packaging | Deduplicate tied weights and package only the manifest-selected 5.659 GiB runtime set |
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
uses interpolated learned patch positions and no vision RoPE.

Phase 3B completed on 2026-08-06. C++ now decodes the original PNG, reproduces
Pillow bicubic resize byte for byte, generates processor-compatible
`pixel_values` and `image_grid_thw`, runs the full ncnn vision chain, computes
text embeddings, and places all 288 visual embeddings at `[2, 290)`. That
generated `[1, 313, 1024]` hidden state feeds the 24-layer prefill and dynamic
decoder directly. Packed and unpacked runs both emit the exact 11 tokens and
`HELLO 2026\nNCNN CPU TEST`.

Phase 3C completed on 2026-08-06. C++ now applies the fixed OCR chat template,
loads the model's exported 120818-entry ByteLevel vocabulary and 119758 BPE
merge ranks, and constructs all 313 input IDs without a captured ID tensor.
Attention masks, multimodal token types, four-axis position IDs, and dynamic
mRoPE are generated in C++ for prefill and every decode step. The discrete
contract fields are exact, mRoPE differs by at most `5.96046e-08`, and packed
and unpacked runs still emit the exact 11-token text through EOS.

### Phase 4: Productize and validate two platforms

1. Create the root CMake project, runtime library, and CLI.
2. Add model-manifest validation and clear runtime diagnostics.
3. Build and run the smoke test on Ubuntu and native Windows/MSVC.
4. Add more images and exact-text regression cases, then measure memory and
   latency.

Phase 4A completed on 2026-08-06. The root CMake project now builds and
installs `HunyuanOCR::runtime`, `hunyuanocr_cli`, and optional independent
parity targets. Product source is free of reference paths. A 162-file,
5.659-GiB manifest supports size and full SHA-256 checks. Packed and unpacked
CLI runs emit the exact 11 tokens and text; measured runtime/peak RSS are
11.717 s/5,978,556 KiB and 11.117 s/4,258,200 KiB, respectively.

### Phase 5: Publish

1. Minimize and document dependencies and model conversion steps.
2. Prepare the repository for public reproduction without development-only
   artifacts.
3. Review and publish the GitHub Discussion with the repository URL.
4. Evaluate a focused ncnn_llm or ncnn pull request for reusable runtime work.

## Next action

Phase 4B is the next highest-priority milestone. Build the installed runtime
and CLI with native Windows/MSVC against the required ncnn changes, run the
same manifest and smoke-image contract, and require the exact 11 tokens and
text. Then record platform-specific build, performance, and memory evidence.
