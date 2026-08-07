# Phase 4I Direct pnnx Runtime Artifact Export Pipeline

## Scope

Phase 4I turns the runtime exporters into staging-aware tools and adds a Linux
pipeline that rebuilds the full runtime artifact set directly into a fresh
persistent staging directory. The pipeline uses the local downloaded
`tencent/HunyuanOCR` HuggingFace model directory, the PyTorch reference tensor
directory for trace/parity inputs, and the local pnnx executable. It does not
copy files from the existing `artifacts/` directory.

Implemented entry point:

```bash
/home/asus/work/hunyuanocr/.venv-reference/bin/python \
  tools/export/rebuild_runtime_artifacts.py \
  --clean-staging
```

Default inputs:

- HuggingFace model: `/home/asus/work/hunyuanocr/models/HunyuanOCR-1.5`
- PyTorch reference tensors: `reference/smoke_en_cpu_fp32`
- Reference Python: `/home/asus/work/hunyuanocr/.venv-reference/bin/python`
- pnnx: `/home/asus/work/hunyuanocr/.venv-pnnx/bin/pnnx`
- Linux OCR CLI: `/home/asus/hunyuanocr-recovery/phase4g/linux-release-build/hunyuanocr_cli`

Default persistent outputs:

- Direct staging artifacts: `/home/asus/hunyuanocr-recovery/phase4i/direct-staging-artifacts`
- Export reports/logs: `/home/asus/hunyuanocr-recovery/phase4i/export-docs`
- Pipeline logs: `/home/asus/hunyuanocr-recovery/phase4i/logs`
- Pipeline report: `docs/phase4i_direct_export_pipeline_report.json`

## Exporter Changes

The runtime exporters now accept staging paths while retaining their historical
defaults:

- `--model-dir`
- `--artifacts-dir`
- `--docs-dir`
- `--reference-dir` where reference tensors are required
- `--pnnx` where pnnx conversion is required

Updated exporters:

- `tools/export/export_text_embedding.py`
- `tools/export/export_lm_head.py`
- `tools/export/export_final_norm.py`
- `tools/export/export_tokenizer_vocab.py`
- `tools/export/export_vision_tower_full.py`
- `tools/export/export_decoder_prefill_kv.py`
- `tools/export/export_decoder_dynamic.py`

The smaller text embedding, LM head, and final RMSNorm exporters now run pnnx
inside the exporter instead of relying on a separate manual pnnx step.

## Direct Export Result

The Phase 4I pipeline ran 30 exporter steps:

- text embedding
- LM head
- final RMSNorm
- tokenizer vocab/merges
- complete vision tower and patch merger
- all 24 decoder prefill KV layers
- all 24 decoder dynamic decode layers

Generated runtime contract:

| Item | Value |
| --- | ---: |
| Runtime files | 170 |
| Total size | 5.724 GiB |
| Manifest structure comparison | passed |
| Digest/size differences vs verified artifacts | 0 |
| Existing artifacts manifest unchanged | true |
| Pipeline elapsed time | 1068.3 s |

The direct staging manifest contains the same relative paths, sizes, and
SHA-256 digests as the current verified `artifacts/runtime_manifest.tsv`.

## Validation

The direct staging directory was validated by the Linux runtime CLI with packed
and unpacked execution.

| Suite | Status | Driver Seconds |
| --- | --- | ---: |
| smoke | passed | 30.4 |
| dynamic | passed | 67.8 |
| real-png | passed | 241.6 |
| real-jpeg | passed | 236.5 |
| cache-budgets | passed | 80.2 |

Coverage:

- smoke OCR output remains exactly `HELLO 2026\nNCNN CPU TEST`
- dynamic image sizes pass packed and unpacked
- real PNG regression passes packed and unpacked
- real JPEG regression passes packed and unpacked
- decoder cache budgets `0`, `512`, and `2048` MiB pass packed and unpacked

## Persistent Evidence

- Pipeline report: `docs/phase4i_direct_export_pipeline_report.json`
- Pipeline log: `/home/asus/hunyuanocr-recovery/phase4i/logs/phase4i_direct_export_pipeline.log`
- Direct staging artifacts: `/home/asus/hunyuanocr-recovery/phase4i/direct-staging-artifacts`
- Export logs: `/home/asus/hunyuanocr-recovery/phase4i/logs/exporters`
- Export reports/pnnx logs: `/home/asus/hunyuanocr-recovery/phase4i/export-docs`
- Smoke validation: `/home/asus/hunyuanocr-recovery/phase4i/logs/validation/smoke/smoke_summary.json`
- Dynamic validation: `/home/asus/hunyuanocr-recovery/phase4i/logs/validation/dynamic/dynamic_summary.json`
- Real PNG validation: `/home/asus/hunyuanocr-recovery/phase4i/logs/validation/real-png/real-png_summary.json`
- Real JPEG validation: `/home/asus/hunyuanocr-recovery/phase4i/logs/validation/real-jpeg/real-jpeg_summary.json`
- Cache budget validation: `/home/asus/hunyuanocr-recovery/phase4i/logs/validation/cache-budgets/cache-budgets_summary.json`

## Remaining Risks

- The direct export still uses the previously captured PyTorch reference tensor
  directory for deterministic trace inputs and parity checks. It no longer uses
  existing converted artifacts as a source.
- The dynamic decoder exporter still runs one process per layer from the
  Phase 4I pipeline. This is simple and robust, but a later optimization could
  load the HF model once and export all dynamic layers in a single process.
- Windows validation was not rerun in Phase 4I. The generated artifacts are
  byte-identical to the already verified runtime artifacts, and Linux
  validation passed, but a later release pass should rerun the Windows package
  matrix after any exporter or pnnx toolchain changes.
