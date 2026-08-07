# Phase 4H Reproducible Runtime Artifact Pipeline

## Scope

Phase 4H productizes a replayable Linux pipeline for assembling the runtime
artifact directory used by `hunyuanocr_cli`. The pipeline starts from the
already downloaded local `tencent/HunyuanOCR` HuggingFace model directory and
the current verified pnnx/ncnn conversion outputs, writes into a separate
persistent staging directory, regenerates manifest metadata, compares the
staging contract against the verified runtime artifacts, and validates OCR
behavior without modifying the existing `artifacts/` directory.

The implemented entry point is:

```bash
python3 tools/export/build_runtime_artifacts.py --clean-staging
```

Default local inputs:

- HuggingFace model: `/home/asus/work/hunyuanocr/models/HunyuanOCR-1.5`
- Verified source artifacts: `/home/asus/work/hunyuanocr/HunyuanOCR-ncnn/artifacts`
- Linux OCR CLI: `/home/asus/hunyuanocr-recovery/phase4g/linux-release-build/hunyuanocr_cli`
- pnnx: `/home/asus/work/hunyuanocr/.venv-pnnx/bin/pnnx`
- ncnn package: `/home/asus/.local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn`

Default persistent outputs:

- Staging artifacts: `/home/asus/hunyuanocr-recovery/phase4h/staging-artifacts`
- Pipeline logs: `/home/asus/hunyuanocr-recovery/phase4h/logs`
- Pipeline report: `docs/phase4h_runtime_artifact_pipeline_report.json`

## Artifact Contract

The runtime manifest generator now supports arbitrary model directories while
preserving its original default behavior for `artifacts/`.

The Phase 4H staging directory contains exactly the runtime contract files:

| Item | Value |
| --- | ---: |
| Runtime files | 170 |
| Total size | 5.724 GiB |
| Manifest comparison | passed |
| Source file count | 170 |
| Staging file count | 170 |

Generated staging metadata:

- `/home/asus/hunyuanocr-recovery/phase4h/staging-artifacts/runtime_manifest.tsv`
- `/home/asus/hunyuanocr-recovery/phase4h/staging-artifacts/runtime_compatibility.tsv`

The staging manifest has the same relative paths, sizes, and SHA-256 digests as
the current verified `artifacts/runtime_manifest.tsv`.

## Validation

The generated staging directory was validated with the Linux Phase 4G release
CLI. Each suite was run with packed and unpacked execution where applicable.

| Suite | Status | Driver Seconds |
| --- | --- | ---: |
| smoke | passed | 25.9 |
| dynamic | passed | 65.5 |
| real-png | passed | 256.8 |
| real-jpeg | passed | 254.3 |
| cache-budgets | passed | 78.7 |

Coverage:

- smoke OCR output remains exactly `HELLO 2026\nNCNN CPU TEST`
- dynamic image sizes pass packed and unpacked
- real PNG regression passes packed and unpacked
- real JPEG regression passes packed and unpacked
- decoder cache budgets `0`, `512`, and `2048` MiB pass packed and unpacked

## Persistent Evidence

- Pipeline report: `docs/phase4h_runtime_artifact_pipeline_report.json`
- Pipeline log: `/home/asus/hunyuanocr-recovery/phase4h/logs/phase4h_runtime_artifacts_pipeline.log`
- Smoke summary: `/home/asus/hunyuanocr-recovery/phase4h/logs/validation/smoke/smoke_summary.json`
- Dynamic summary: `/home/asus/hunyuanocr-recovery/phase4h/logs/validation/dynamic/dynamic_summary.json`
- Real PNG summary: `/home/asus/hunyuanocr-recovery/phase4h/logs/validation/real-png/real-png_summary.json`
- Real JPEG summary: `/home/asus/hunyuanocr-recovery/phase4h/logs/validation/real-jpeg/real-jpeg_summary.json`
- Cache budget summary: `/home/asus/hunyuanocr-recovery/phase4h/logs/validation/cache-budgets/cache-budgets_summary.json`

## Remaining Risks

- Phase 4H assembles a clean runtime artifact staging directory from verified
  local conversion outputs. The lower-level pnnx exporter scripts are still
  mostly component-oriented and keep their historical default output path under
  `artifacts/`.
- A later pipeline hardening pass should add isolated per-component pnnx export
  work directories and direct staging output for every exporter, so a single
  command can rebuild each `.ncnn.param` and `.ncnn.bin` file from model weights
  without relying on the existing converted source artifact directory.
- Windows validation was not rerun in Phase 4H; this phase is the Linux
  artifact-generation pipeline requested for staging. Existing Windows package
  and runtime validation remains recorded in Phase 4G.
