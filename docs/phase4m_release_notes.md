# HunyuanOCR-ncnn Release Candidate Notes

## Summary

This release candidate provides a CPU-only ncnn runtime and OCR CLI for
HunyuanOCR-1.5. The source tree contains the runtime, CMake package, tests,
conversion/export scripts, release validation scripts, and documentation. The
converted model files remain external to the binary packages.

## What Is Included

- C++17 runtime library exported as `HunyuanOCR::runtime`
- OCR CLI with Linux and native Windows/MSVC support
- PNG and JPEG image input
- Dynamic processor-compatible image grids
- Vision tower, patch merger, prompt construction, 24-layer prefill, decoder
  KV cache, greedy generation, and tokenizer decode
- Runtime manifest and compatibility validation
- Budgeted decoder layer cache for lower long-output latency
- CTest release suites for smoke, dynamic-size, real PNG, real JPEG, cache
  budgets, and error paths
- Reproducible artifact pipeline from a local `tencent/HunyuanOCR`
  HuggingFace model directory

## What Is Not Included

- HunyuanOCR model weights
- Converted ncnn model artifacts inside the binary packages
- A published GitHub Discussion
- An upstream ncnn_llm pull request

## Release Evidence

The final pre-release gate is Phase 4L:

- Report: `docs/phase4l_dual_platform_release_acceptance.json`
- Milestone: `docs/phase4l_dual_platform_release_acceptance_milestone.md`
- Phase 4K reproduced runtime manifest SHA-256:
  `71498acaeafff31e2cbfa4c3ed9de81b73d9078e1f2bc7528e87bc36d7222431`
- Phase 4K reproduced runtime compatibility SHA-256:
  `cc47674acdbd3770952294b9363952fbea347acbeb355a7d363bd7e6c86c73f6`

Linux package validation:

- Model source: `/home/asus/hunyuanocr-recovery/phase4k/direct-staging-artifacts`
- CTest suites passed: smoke, dynamic, real-png, real-jpeg, cache-budgets,
  error-paths
- Packages produced: TGZ and ZIP
- Extracted package OCR passed in packed and unpacked modes

Windows package validation:

- Model source: manifest-selected NTFS copy at
  `D:\hunyuanocr-recovery\phase4l\model-ntfs`
- Manifest-selected files copied and SHA-256 checked: 170
- CTest suites passed: smoke, dynamic, real-png, real-jpeg, cache-budgets,
  error-paths
- Package produced: ZIP
- Extracted package OCR passed in packed and unpacked modes

## License And Model Notes

The project source, runtime, CLI, tests, scripts, and documentation are
licensed under Apache-2.0. Third-party notices are collected in
`THIRD_PARTY_NOTICES.md` and `third_party/licenses/`.

Tencent HunyuanOCR model files are not licensed under Apache-2.0. Users must
review and comply with the Tencent Hunyuan Community License Agreement before
downloading, converting, distributing, or using the model files.

## Final Manual Steps Before Public Release

1. Add the final GitHub remote.
2. Push the release branch and milestone tags.
3. Attach or archive the Linux TGZ/ZIP and Windows ZIP packages externally.
4. Replace the repository URL placeholder in
   `docs/github_discussion_hunyuanocr_ncnn_draft.md`.
5. Publish the GitHub Discussion manually after a final license wording review.
6. Decide whether to prepare an optional ncnn_llm upstream pull request.
