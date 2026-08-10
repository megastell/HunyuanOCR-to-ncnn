# Draft GitHub Discussion: HunyuanOCR-ncnn CPU Runtime

Repository URL: https://github.com/megastell/HunyuanOCR-to-ncnn
Release URL: https://github.com/megastell/HunyuanOCR-to-ncnn/releases/tag/v0.1.0

## Title

Porting HunyuanOCR-1.5 to ncnn with a reproducible CPU-only C++ runtime

## Post

I have been working on a CPU-only ncnn port of HunyuanOCR-1.5, and the first public repository and binary runtime packages are now available:

https://github.com/megastell/HunyuanOCR-to-ncnn

The v0.1.0 release is here:

https://github.com/megastell/HunyuanOCR-to-ncnn/releases/tag/v0.1.0

The short version: the repository contains a C++17 OCR runtime, a CLI, CMake install/export support, pnnx/ncnn export scripts, PyTorch reference capture scripts, and Linux plus native Windows validation. The runtime packages do not include HunyuanOCR model weights or converted ncnn artifacts. Users need to obtain the original `tencent/HunyuanOCR` model themselves, follow the model license, and generate or provide a local runtime artifact directory.

The main goal was not just to make the model run. I wanted an end-to-end path where the ncnn runtime produces the same final OCR text as the PyTorch reference on the same inputs, with every large generated artifact reproducible from a local HuggingFace model directory.

## What the runtime does

The production path no longer loads captured prompt tensors or captured prefill hidden states. It builds the fixed OCR prompt in C++, constructs tokenizer inputs, image token placement, dynamic image grids, multimodal position ids, mRoPE data, and image pixels, then runs the model through ncnn.

The actual inference path is:

1. Decode PNG/JPEG input with the C++ image path.
2. Build `pixel_values` and `image_grid_thw`.
3. Run the ncnn vision tower and patch merger.
4. Place the vision embedding into the image-token span.
5. Run the full 24-layer decoder prefill.
6. Continue with an autoregressive decode loop and KV cache.
7. Run final RMSNorm, LM head, greedy token selection, and tokenizer decode.

Decoder layers can be streamed from disk, or cached under a memory budget with `--decoder-cache-mib`. That was important because loading every decoder network permanently is simple, but expensive. The budgeted cache gives a practical tradeoff between memory and long-output latency.

## How parity was built

The project was developed as a chain of small parity gates. I started from pieces that were easy to isolate, then moved the boundary outward:

- token embedding and LM head
- decoder Layer 0 prefill/decode
- all 24 decoder layers
- dynamic decode and KV cache
- vision tower and patch merger
- C++ image preprocessing
- prompt/tokenizer/mRoPE construction
- full OCR generation on smoke, dynamic-size, real PNG, and real JPEG inputs

The most useful rule was to stop treating "close enough" as a final answer. When a boundary failed, I captured the exact PyTorch tensor at that boundary, compared it with the ncnn output, and only moved on once the difference was explained.

JPEG needed one extra bit of care. Pillow/libjpeg and stb_image can produce slightly different pixels from the same JPEG, which is enough to break exact token parity. The current validation uses an explicit pixel contract: the production C++ path exports the exact stb_image RGB pixels, and the PyTorch reference consumes those same pixels losslessly. With that, Linux and Windows both reproduce the reference tokens for the real JPEG cases.

## Reproducible artifacts

The release validation uses runtime artifacts regenerated from a local `tencent/HunyuanOCR` HuggingFace directory, not checked-in converted artifacts.

The Linux reproduction gate regenerates PyTorch references, exports the ncnn runtime artifacts into a clean staging directory, writes manifest/compatibility metadata, and runs validation:

```bash
$HOME/work/hunyuanocr/.venv-reference/bin/python \
  tools/release/reproduce_runtime_artifacts_acceptance.py \
  --clean-staging \
  --hf-model-dir "$HOME/work/hunyuanocr/models/HunyuanOCR-1.5" \
  --work-dir "$HOME/hunyuanocr-recovery/phase4k"
```

The same path is also registered through CTest:

```bash
cmake -S . -B build-phase4k \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$HOME/.local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn" \
  -DHUNYUANOCR_ENABLE_REPRODUCTION_TESTS=ON \
  -DHUNYUANOCR_REPRO_HF_MODEL_DIR="$HOME/work/hunyuanocr/models/HunyuanOCR-1.5" \
  -DHUNYUANOCR_REPRO_WORK_DIR="$HOME/hunyuanocr-recovery/phase4k" \
  -DHUNYUANOCR_REPRO_REFERENCE_PYTHON="$HOME/work/hunyuanocr/.venv-reference/bin/python" \
  -DHUNYUANOCR_REPRO_PNNX="$HOME/work/hunyuanocr/.venv-pnnx/bin/pnnx"
cmake --build build-phase4k --parallel
ctest --test-dir build-phase4k -L reproducible-artifacts --output-on-failure
```

For the current release gate, the regenerated runtime artifact directory had 170 files and was about 5.724 GiB. The important metadata hashes were:

```text
runtime_manifest.tsv       71498acaeafff31e2cbfa4c3ed9de81b73d9078e1f2bc7528e87bc36d7222431
runtime_compatibility.tsv  cc47674acdbd3770952294b9363952fbea347acbeb355a7d363bd7e6c86c73f6
```

## Release validation

Before publishing v0.1.0, I ran the Linux and native Windows/MSVC release gates against the reproduced artifacts. The test set covers smoke OCR, dynamic image sizes, real PNG, real JPEG, cache-budget behavior, and error paths.

The Linux TGZ/ZIP packages and the Windows ZIP package were extracted into clean directories and run in both packed and unpacked modes. The Windows validation copies only manifest-selected model files to NTFS and verifies them by SHA-256 before running the installed CLI.

Release package hashes:

```text
4315e56640357b44410b449be763d661f8444f9de9476bd907ff3d713c4e9290  HunyuanOCR-ncnn-0.1.0-Linux-x86_64.tar.gz
247d67bc6b3200729f58c688aeb1d9da10dc2aa876f0a3e08122eafd211df18f  HunyuanOCR-ncnn-0.1.0-Linux-x86_64.zip
7af8b49f511deb08a502b01fc7ece91353fc2841ee72996ad8369999a6ad914f  HunyuanOCR-ncnn-0.1.0-Windows-AMD64.zip
```

## License and model files

The source code, runtime library, CLI, CMake package, tests, and project documentation are Apache-2.0.

The HunyuanOCR model files are separate. They are governed by the Tencent Hunyuan Community License Agreement, not by this repository's Apache-2.0 license. The release packages intentionally do not include model weights, HuggingFace files, PyTorch reference tensors, or converted ncnn runtime artifacts.

Please review the Tencent model license before downloading, converting, distributing, or using the model files. This project is not affiliated with, sponsored by, or endorsed by Tencent.

## What is next

The runtime is usable now, but there are still useful directions left: more hardware and OS coverage, lower memory use for large images and long generations, better documentation around artifact generation failures, and possibly extracting the reusable LLM/runtime pieces into an upstream ncnn_llm pull request.

For now, I am publishing the repo and the first release so other people can inspect the approach, reproduce the artifacts locally, and try the CPU runtime on their own OCR images.
