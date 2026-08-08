# Draft GitHub Discussion: HunyuanOCR-ncnn CPU Runtime

Repository URL: TODO replace with the public HunyuanOCR-ncnn repository URL.

## Title

Porting HunyuanOCR-1.5 to ncnn with a reproducible CPU-only C++ runtime

## Post

This is a technical summary of a HunyuanOCR-1.5 to ncnn port. The goal was to
run OCR end to end in C++ with a small runtime dependency surface, while keeping
the final generated text aligned with the original PyTorch model on the same
inputs.

The repository contains:

- C++17 runtime library and OCR CLI
- CMake install/export package as `HunyuanOCR::runtime`
- pnnx/ncnn exporters for text embedding, vision tower, patch merger, decoder
  prefill/decode, final norm, LM head, tokenizer vocabulary, manifest, and
  compatibility metadata
- Reference capture scripts for PyTorch CPU FP32 parity
- Linux and native Windows/MSVC release validation scripts
- Reproducible artifact generation from a local `tencent/HunyuanOCR`
  HuggingFace model directory

The converted model files are intentionally not bundled in the runtime binary
packages. They are external artifacts governed by the Tencent Hunyuan Community
License Agreement. Runtime startup validates the model directory through
`runtime_manifest.tsv` and, when present, `runtime_compatibility.tsv`.

## Runtime Path

The production path no longer loads captured prompt or prefill tensors. It
constructs the fixed OCR prompt, tokenizer inputs, multimodal positions, image
grid, image token span, mRoPE data, and pixel preprocessing in C++. The runtime
then executes:

1. Image preprocessing for PNG/JPEG.
2. ncnn vision tower and patch merger.
3. Multimodal embedding placement.
4. 24-layer decoder prefill.
5. Autoregressive decoder loop with KV cache.
6. Final norm, LM head, greedy token selection, and tokenizer decode.

Decoder model files can be streamed layer by layer, or partially retained in
memory through `--decoder-cache-mib`. This keeps the default memory footprint
lower while still allowing users to trade memory for long-output latency.

## Parity Strategy

The project was developed by progressively tightening boundaries:

- token embedding and LM head
- decoder Layer 0 prefill/decode
- 24-layer prefill
- dynamic decoder KV generation
- vision tower and patch merger
- C++ image preprocessing and multimodal prompt construction
- full OCR generation for smoke, dynamic-size, real PNG, and real JPEG cases

JPEG parity uses an explicit pixel contract: the same stb_image RGB pixels used
by production are exported and fed to the PyTorch reference path. This avoids
cross-decoder drift between Pillow/libjpeg and stb_image.

## Release Validation

The latest local release gate used runtime artifacts reproduced from a local
HuggingFace model directory, not pre-existing repository artifacts.

Phase 4K regenerated references and directly exported runtime artifacts:

- Manifest file count: 170
- Runtime artifact size: 5.724 GiB
- Manifest SHA-256:
  `71498acaeafff31e2cbfa4c3ed9de81b73d9078e1f2bc7528e87bc36d7222431`
- Compatibility SHA-256:
  `cc47674acdbd3770952294b9363952fbea347acbeb355a7d363bd7e6c86c73f6`

Phase 4L validated the reproduced artifacts on both platforms:

- Linux CTest suites: smoke, dynamic, real-png, real-jpeg, cache-budgets,
  error-paths
- Windows/MSVC CTest suites: smoke, dynamic, real-png, real-jpeg,
  cache-budgets, error-paths
- Linux TGZ/ZIP package extraction and OCR passed in packed and unpacked modes
- Windows ZIP package extraction and OCR passed in packed and unpacked modes
- Windows model directory was copied to NTFS using only manifest-selected files
  and verified by SHA-256

## Reproduce From A Local HF Model

After downloading the HunyuanOCR model separately, the Linux reproduction gate
can regenerate PyTorch references, export ncnn runtime artifacts into a clean
staging directory, generate manifest/compatibility metadata, and run Linux
validation:

```bash
$HOME/work/hunyuanocr/.venv-reference/bin/python \
  tools/release/reproduce_runtime_artifacts_acceptance.py \
  --clean-staging \
  --hf-model-dir "$HOME/work/hunyuanocr/models/HunyuanOCR-1.5" \
  --work-dir "$HOME/hunyuanocr-recovery/phase4k"
```

The release package gate then uses the reproduced staging artifacts:

```bash
bash tools/release/run_phase4l_linux_release_acceptance.sh
```

Native Windows/MSVC validation copies those reproduced artifacts to NTFS and
runs the Windows release gate:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\windows\validate_phase4l_msvc.ps1
```

## Notes

The project is CPU-focused and intentionally conservative about runtime
dependencies. The source/runtime code is Apache-2.0, while the HunyuanOCR model
files remain under Tencent's model license. Please review the model license
before distributing converted artifacts.

The optional next step is to extract generally useful pieces into an upstream
ncnn_llm pull request after the public repository is available.
