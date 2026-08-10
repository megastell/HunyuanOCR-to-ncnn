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

---

## 中文版本

这段时间我在做一件比较具体的事：把 HunyuanOCR-1.5 移植到 ncnn，并且做成一个 CPU-only 的 C++ runtime。现在第一版公开仓库和二进制运行包已经放出来了：

https://github.com/megastell/HunyuanOCR-to-ncnn

v0.1.0 Release 在这里：

https://github.com/megastell/HunyuanOCR-to-ncnn/releases/tag/v0.1.0

这个仓库里主要包括 C++17 OCR runtime、命令行工具、CMake install/export、pnnx/ncnn 导出脚本、PyTorch 参考捕获脚本，以及 Linux 和原生 Windows/MSVC 的验证脚本。

Release 包里没有放 HunyuanOCR 权重，也没有放转换后的 ncnn 模型 artifacts。模型需要用户自己从 `tencent/HunyuanOCR` 获取，并遵守腾讯混元模型许可证；转换后的 runtime artifacts 也需要用户在本地生成，或者提供自己已经校验过的 artifacts 目录。

我做这个项目的目标不是“能跑起来就行”，而是希望在相同输入下，ncnn runtime 最终输出的 OCR 文本能和 PyTorch 参考结果对齐，同时整个大模型转换过程可以从本地 HuggingFace 模型目录重新复现。

## 现在的运行路径

正式 runtime 已经不再加载捕获好的 prompt tensor 或 prefill hidden state。它会在 C++ 里构造固定 OCR prompt、tokenizer 输入、图片 token 放置位置、动态 image grid、多模态 position ids、mRoPE 数据和图片像素，然后交给 ncnn 执行。

实际推理链路大概是这样：

1. C++ 读取 PNG/JPEG 图片。
2. 生成 `pixel_values` 和 `image_grid_thw`。
3. 执行 ncnn vision tower 和 patch merger。
4. 把视觉 embedding 放到 prompt 中对应的图片 token span。
5. 执行完整 24 层 decoder prefill。
6. 进入带 KV cache 的自回归 decode loop。
7. 执行 final RMSNorm、LM head、greedy token 选择和 tokenizer decode。

decoder 层可以按层从磁盘流式加载，也可以通过 `--decoder-cache-mib` 设置一个内存预算，把一部分 decoder 模型 bytes 缓存在内存里。这样默认内存占用不会太夸张，同时长文本生成时也可以用更多内存换一些速度。

## 对齐是怎么做的

这个项目基本是靠一段一段对齐推进出来的。先从最容易隔离的小模块开始，再逐步把边界往外扩：

- token embedding 和 LM head
- decoder Layer 0 prefill/decode
- 完整 24 层 decoder
- 动态 decode 和 KV cache
- vision tower 和 patch merger
- C++ 图片预处理
- prompt/tokenizer/mRoPE 构造
- smoke、动态尺寸、真实 PNG、真实 JPEG 的完整 OCR 生成

最有用的经验是：不要把“差不多”当成最终答案。每次边界不对，就捕获 PyTorch 在同一边界上的参考 tensor，和 ncnn 输出直接比较；只有差异能解释清楚，才继续往下一段走。

JPEG 这里额外踩了一个坑。Pillow/libjpeg 和 stb_image 对同一张 JPEG 解码出来的像素可能有细微差异，而这种差异已经足够影响最终 token。现在的验证方式是明确像素契约：生产路径使用 stb_image，PyTorch 参考也吃 C++ 导出的同一份 RGB 像素。这样 Linux 和 Windows 的真实 JPEG case 都能复现参考 token。

## artifacts 复现

当前 Release 验收用到的 artifacts 不是仓库里预先放好的转换产物，而是从本地 `tencent/HunyuanOCR` HuggingFace 模型目录重新生成的。

Linux 下可以用这条命令重新生成 PyTorch references、导出 ncnn runtime artifacts、生成 manifest/compatibility，并跑验证：

```bash
$HOME/work/hunyuanocr/.venv-reference/bin/python \
  tools/release/reproduce_runtime_artifacts_acceptance.py \
  --clean-staging \
  --hf-model-dir "$HOME/work/hunyuanocr/models/HunyuanOCR-1.5" \
  --work-dir "$HOME/hunyuanocr-recovery/phase4k"
```

同一条复现链也接进了 CTest：

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

这次 Release gate 里重新生成出来的 runtime artifacts 是 170 个文件，总大小约 5.724 GiB。关键元数据 hash 是：

```text
runtime_manifest.tsv       71498acaeafff31e2cbfa4c3ed9de81b73d9078e1f2bc7528e87bc36d7222431
runtime_compatibility.tsv  cc47674acdbd3770952294b9363952fbea347acbeb355a7d363bd7e6c86c73f6
```

## Release 验证

v0.1.0 发布前，我用重新生成的 artifacts 跑了 Linux 和原生 Windows/MSVC 的 release gate。测试覆盖 smoke OCR、动态图片尺寸、真实 PNG、真实 JPEG、decoder cache budget 和错误路径。

Linux TGZ/ZIP 和 Windows ZIP 都做了解包到干净目录后的运行验证，并且 packed/unpacked 两种模式都跑过。Windows 侧会把 manifest 选中的模型文件复制到 NTFS，再做 SHA-256 校验，然后运行安装后的 CLI。

Release 包 hash：

```text
4315e56640357b44410b449be763d661f8444f9de9476bd907ff3d713c4e9290  HunyuanOCR-ncnn-0.1.0-Linux-x86_64.tar.gz
247d67bc6b3200729f58c688aeb1d9da10dc2aa876f0a3e08122eafd211df18f  HunyuanOCR-ncnn-0.1.0-Linux-x86_64.zip
7af8b49f511deb08a502b01fc7ece91353fc2841ee72996ad8369999a6ad914f  HunyuanOCR-ncnn-0.1.0-Windows-AMD64.zip
```

## 许可证边界

这个仓库里的源码、runtime、CLI、CMake package、测试和项目文档使用 Apache-2.0。

HunyuanOCR 模型文件是另一回事。它们受 Tencent Hunyuan Community License Agreement 约束，不属于本仓库 Apache-2.0 的覆盖范围。Release 包也刻意没有包含模型权重、HuggingFace 文件、PyTorch reference tensors 或转换后的 ncnn runtime artifacts。

如果要下载、转换、分发或使用模型文件，请先认真看腾讯混元模型许可证。这个项目和腾讯没有从属、赞助或背书关系。

## 后面还可以做什么

目前 runtime 已经可以用，但还可以继续往前走：更多硬件和系统覆盖、更低的大图/长文本内存占用、更清楚的 artifacts 生成失败诊断，以及把通用的 LLM runtime 部分整理出来，之后看是否适合给 ncnn_llm 提 PR。

先把仓库和第一版 Release 发出来，是希望大家可以看到完整实现方式，在本地复现 artifacts，也可以拿自己的 OCR 图片试一下这个 CPU runtime。
