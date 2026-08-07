# Third-Party Notices

This file summarizes the third-party components and external model licenses
that matter for source and binary releases of HunyuanOCR-ncnn. License copies
are archived in `third_party/licenses/`.

## Runtime Dependencies

| Component | Use | License | Local License Copy |
| --- | --- | --- | --- |
| ncnn | CPU inference runtime and CMake dependency | BSD 3-Clause with bundled third-party notices | `third_party/licenses/ncnn-LICENSE.txt` |
| stb_image | PNG/JPEG image decoding header | Public domain | `third_party/licenses/stb_image-LICENSE.txt` |

The binary release links ncnn from the configured local ncnn installation.
Release validation records the resulting platform dependencies in
`docs/linux_phase4g_release_dryrun.json` and
`docs/windows_phase4g_release_dryrun.json`.

## Model And Reference Materials

| Component | Use | License | Local License Copy |
| --- | --- | --- | --- |
| Tencent HunyuanOCR | Source model for conversion and runtime OCR inference | Tencent Hunyuan Community License Agreement | `third_party/licenses/Tencent-HunyuanOCR-LICENSE.txt` |

The converted ncnn model files are not part of the source-code license and are
not bundled in the runtime binary packages. Users must obtain and use
HunyuanOCR model files only in compliance with the Tencent Hunyuan Community
License Agreement, including its territory, acceptable-use, distribution, and
notice requirements.

Required notice for distributions of Tencent Hunyuan Works:

> Tencent Hunyuan is licensed under the Tencent Hunyuan Community License
> Agreement, Copyright (C) 2025 Tencent. All Rights Reserved. The trademark
> rights of "Tencent Hunyuan" are owned by Tencent or its affiliate.

This project is not affiliated with, sponsored by, or endorsed by Tencent.

## Project License

The HunyuanOCR-ncnn source code, runtime library, CLI, CMake packaging, tests,
and project documentation are licensed under Apache-2.0 unless a file states
otherwise.
