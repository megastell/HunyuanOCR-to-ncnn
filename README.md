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
