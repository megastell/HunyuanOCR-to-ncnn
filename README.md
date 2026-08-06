# HunyuanOCR-ncnn

HunyuanOCR-1.5 的 CPU-only ncnn 移植、严格精度对齐与
Windows/Linux 双平台部署项目。

## 项目目标

- 使用 pnnx 转换 HunyuanOCR-1.5
- 使用 C++17 实现端到端 OCR 推理
- 部署运行时尽量只依赖 ncnn 和 C++ 标准库
- PyTorch 与 ncnn 最终输出文本严格对齐
- 支持 Ubuntu 24.04 与 Windows x64
- 提供低内存分模块转换和自动对拍工具

## 当前进度

- [x] Ubuntu 24.04 WSL2 环境
- [x] ncnn CPU 构建
- [x] PyTorch CPU 环境
- [x] pnnx 安装
- [x] 最小模型完整转换链路
- [ ] HunyuanOCR 模型下载与版本固定
- [ ] PyTorch CPU 确定性基准
- [ ] Vision 模块转换
- [ ] Text embedding 转换
- [ ] Decoder 与 KV-cache 转换
- [ ] LM head 转换
- [ ] C++ 图像预处理
- [ ] C++ tokenizer
- [ ] C++ 自回归生成
- [ ] Linux 真实模型验证
- [ ] Windows 真实模型验证

## 基线配置

- HunyuanOCR: 1.5 base AR
- Runtime: ncnn CPU
- Reference: PyTorch CPU FP32
- Decoding: greedy
- Build: CMake
- Platforms: Ubuntu 24.04 / Windows x64

## Phase 4A Runtime

The repository now has a reusable C++17 runtime library and OCR CLI. The
runtime accepts the converted model directory and a PNG image, and does not
load any captured PyTorch reference tensor.

```bash
cmake -S . -B build-runtime -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR="$HOME/.local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn"
cmake --build build-runtime --parallel

build-runtime/hunyuanocr_cli \
  --model-dir "$PWD/artifacts" \
  --image "$PWD/tests/assets/ocr_smoke_en.png" \
  --packing 1 \
  --threads 9 \
  --verify size
```

Generate or refresh `artifacts/runtime_manifest.tsv` after conversion with:

```bash
$HOME/work/hunyuanocr/.venv-reference/bin/python \
  tools/export/export_runtime_manifest.py
```

`--verify size` checks the complete 162-file inventory at normal startup.
`--verify sha256` performs a full content hash check for installation or
release validation. The current public runtime contract uses the fixed OCR
prompt and the audited `[1,22,50]` image grid. Reference-backed parity tests
remain available through `-DHUNYUANOCR_BUILD_PARITY_TESTS=ON`.
