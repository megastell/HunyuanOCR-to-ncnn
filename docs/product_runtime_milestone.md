# Product Runtime and OCR CLI Milestone

## Scope

Phase 4A turns the proven image-to-text test chain into a reusable C++17
runtime and command-line application. The product path starts from a PNG and a
converted model directory and does not load captured PyTorch tensors. The
reference-backed executable remains a separate optional parity target.

## Runtime Layout

- Public API: `include/hunyuanocr/runtime.h`.
- Runtime implementation: `src/`.
- OCR application: `app/hunyuanocr_cli.cpp`.
- Root build and install project: `CMakeLists.txt`.
- Installed CMake package target: `HunyuanOCR::runtime`.
- Independent installed-package consumer: `tests/runtime_api_consumer`.
- Optional parity build: `-DHUNYUANOCR_BUILD_PARITY_TESTS=ON`.

`hunyuanocr::Runtime` owns model loading and exposes `recognize()`. Its options
select packed or unpacked ncnn execution, CPU thread count, token limit, and
manifest verification mode. Results contain token IDs, decoded UTF-8 text,
image dimensions, EOS state, and phase timings.

## CLI Contract

The CLI accepts:

```text
--model-dir <artifacts>
--image <png>
--packing 0|1
--threads <positive integer>
--max-new-tokens <positive integer>
--verify none|size|sha256
```

Unpacked execution is the default because it was both faster and lower-memory
on the audited CPU. The fixed Phase 3C OCR prompt and `[1,22,50]` image grid
remain the current public inference contract.

## Model Manifest

`tools/export/export_runtime_manifest.py` selects only the product model set
from the 44 GiB development artifact tree and writes
`artifacts/runtime_manifest.tsv` plus `docs/runtime_manifest.json`.

- Runtime files: 162.
- Runtime model set: 6,076,349,856 bytes (`5.659 GiB`).
- Vision models: 1,758,571,098 bytes.
- 24 prefill models: 1,661,294,112 bytes.
- 24 dynamic decoder models: 1,661,299,680 bytes.
- Text embedding: 494,870,697 bytes.
- LM head: 494,870,690 bytes.
- Tokenizer assets: 5,439,325 bytes.

Normal startup verifies the full inventory and file sizes. Release validation
can select full SHA-256 verification. The C++ SHA implementation matched all
162 hashes generated independently by Python.

## Separation From Parity

A source audit over `include/`, `src/`, and `app/` finds no reference path or
captured tensor filename. The root CMake parity option builds the existing
`multimodal_full_generation` and `prompt_inputs_contract` executables in a
separate target tree. Those tests may load references; the runtime library and
CLI do not.

## Exact Output

Packed and unpacked product CLI runs both emit:

```text
93892 5112 206 1717 21 185 18009 15613 16678 21836 120007
```

Both reach EOS and decode exactly to:

```text
HELLO 2026
NCNN CPU TEST
```

The independent packed and unpacked parity modes produce the same exact result
and retain the established layer, KV, RMSNorm, and logits checks.

## Linux Measurements

The clean Phase 4A validation used Ubuntu 24.04 WSL2, 9 CPU threads, Release
builds, and size-manifest verification.

| Mode | Load | Input | Prefill | Decode | Runtime | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Packed | 0.360 s | 3.409 s | 1.483 s | 6.825 s | 11.717 s | 5,978,556 KiB |
| Unpacked | 0.535 s | 7.695 s | 1.318 s | 2.104 s | 11.117 s | 4,258,200 KiB |

Full SHA-256 verification added approximately 16.8 seconds to model loading on
this filesystem. It is therefore an installation check, not the normal
per-invocation mode.

## Memory Optimization Plan

1. Add a low-memory decoder residency mode. The current runtime keeps all 24
   dynamic decoder models resident during generation; their files total
   `1.547 GiB`. Loading one decoder per layer and token can trade model reload
   time for a material peak-memory reduction. Measure that trade before making
   it the default.
2. Share tied embedding/LM-head storage. Their `494,870,532`-byte ncnn weight
   files are byte-identical (SHA-256
   `ddfe020dcafa57b5374992f39399ce3f2cc76497f80f92f9c33724e7c39389e1`).
   A shared read-only weight owner or custom tied-weight layers can avoid the
   second in-memory copy and also deduplicate release packaging.
3. Keep unpacked execution as the CPU default for this baseline. It reduced
   measured peak RSS by about `1.64 GiB` and total runtime by about `0.60 s`.
4. Package only the manifest-selected 5.659 GiB runtime set. The 44 GiB
   development artifact tree and all reference outputs should remain outside
   release archives.

## Persistent Evidence

- Validation report: `docs/product_runtime_validation.json`.
- Manifest report: `docs/runtime_manifest.json`.
- Reproduction script: `tools/validate/validate_product_runtime.sh`.
- Configure, build, install, CLI, SHA, and parity logs:
  `~/hunyuanocr-recovery/phase4a/`.

## Remaining Boundary

Native Windows/MSVC is validated in the follow-up Phase 4B milestone documented
in `docs/windows_msvc_milestone.md`. The runtime still supports the fixed OCR
prompt and smoke image grid rather than arbitrary processor shapes and chat
turns. The two local ncnn precision changes also need a reproducible release
carry or upstream resolution.
