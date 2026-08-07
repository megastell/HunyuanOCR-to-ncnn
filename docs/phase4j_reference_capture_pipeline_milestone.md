# Phase 4J Reference Capture And Direct Export Pipeline

## Scope

Phase 4J adds a reproducible PyTorch reference-capture stage in front of the
Phase 4I direct pnnx export pipeline. The full Linux pipeline now starts from
the local `tencent/HunyuanOCR` HuggingFace model directory, the original smoke
and regression images, and the fixed OCR prompt. It regenerates the exporter
reference tensors into a new persistent staging directory, then invokes the
direct pnnx artifact rebuild and Linux validation.

Implemented entry point:

```bash
/home/asus/work/hunyuanocr/.venv-reference/bin/python \
  tools/reference/rebuild_references_and_artifacts.py \
  --clean-staging
```

Default inputs:

- HuggingFace model: `/home/asus/work/hunyuanocr/models/HunyuanOCR-1.5`
- Smoke image: `tests/assets/ocr_smoke_en.png`
- Dynamic images:
  - `tests/assets/ocr_wide_en.png`
  - `tests/assets/ocr_square_en.png`
  - `tests/assets/ocr_tall_en.png`
- Real OCR images:
  - `tests/assets/ocr_receipt_real.png`
  - `tests/assets/ocr_receipt_real.jpg`
  - `tests/assets/ocr_receipt_real_stb.ppm`
- Reference Python: `/home/asus/work/hunyuanocr/.venv-reference/bin/python`
- pnnx: `/home/asus/work/hunyuanocr/.venv-pnnx/bin/pnnx`
- Linux OCR CLI: `/home/asus/hunyuanocr-recovery/phase4g/linux-release-build/hunyuanocr_cli`

Default persistent outputs:

- Smoke/exporter references: `/home/asus/hunyuanocr-recovery/phase4j/reference-smoke-cpu-fp32`
- Reference raw artifacts: `/home/asus/hunyuanocr-recovery/phase4j/reference-artifacts`
- Reference reports: `/home/asus/hunyuanocr-recovery/phase4j/reference-docs`
- Direct export artifacts: `/home/asus/hunyuanocr-recovery/phase4j/direct-staging-artifacts`
- Direct export reports: `/home/asus/hunyuanocr-recovery/phase4j/direct-export-docs`
- Logs: `/home/asus/hunyuanocr-recovery/phase4j/logs`

## Capture Coverage

The Phase 4J reference capture pipeline regenerates:

- top-level smoke `input_ids`, `attention_mask`, `mm_token_type_ids`,
  `pixel_values`, and `image_grid_thw`
- deterministic smoke generated token IDs and output text
- split contract tensors for text embedding, vision outputs, final RMSNorm,
  LM head, logits, and KV cache summary
- full vision tower boundaries:
  - patch embedding
  - all 27 vision blocks
  - patch merger input/output
- decoder dynamic decode references for steps 1, 2, and 3
- all 24 decoder prefill KV references
- dynamic image expected outputs
- real PNG/JPEG expected outputs, including the JPEG stb RGB pixel contract

Capture result:

| Item | Value |
| --- | ---: |
| Capture steps | 7 |
| Smoke reference `.npy` files | 1255 |
| Smoke text | `HELLO 2026\nNCNN CPU TEST` |
| Smoke token IDs | `[93892, 5112, 206, 1717, 21, 185, 18009, 15613, 16678, 21836, 120007]` |
| Pipeline elapsed time | 1218.9 s |

## Direct Export Result

The Phase 4I direct pnnx export pipeline was rerun using the newly captured
Phase 4J reference directory rather than the repository's existing
`reference/smoke_en_cpu_fp32` directory.

| Item | Value |
| --- | ---: |
| Runtime files | 170 |
| Total size | 5.724 GiB |
| Manifest structure comparison | passed |
| Digest/size differences vs verified artifacts | 0 |
| Existing artifacts manifest unchanged | true |

## Linux Validation

The Phase 4J direct staging artifacts were validated by the Linux runtime CLI
with packed and unpacked execution.

| Suite | Status | Driver Seconds |
| --- | --- | ---: |
| smoke | passed | 28.2 |
| dynamic | passed | 65.3 |
| real-png | passed | 232.8 |
| real-jpeg | passed | 259.7 |
| cache-budgets | passed | 82.0 |

Coverage:

- smoke OCR output remains exactly `HELLO 2026\nNCNN CPU TEST`
- dynamic image sizes pass packed and unpacked
- real PNG regression passes packed and unpacked
- real JPEG regression passes packed and unpacked
- decoder cache budgets `0`, `512`, and `2048` MiB pass packed and unpacked

## Persistent Evidence

- Phase 4J pipeline report: `docs/phase4j_reference_capture_pipeline_report.json`
- Phase 4J direct export report: `docs/phase4j_direct_export_pipeline_report.json`
- Phase 4J pipeline log: `/home/asus/hunyuanocr-recovery/phase4j/logs/phase4j_reference_capture_pipeline.log`
- Reference capture logs: `/home/asus/hunyuanocr-recovery/phase4j/logs/reference-capture`
- Reference reports: `/home/asus/hunyuanocr-recovery/phase4j/reference-docs`
- Direct export logs: `/home/asus/hunyuanocr-recovery/phase4j/logs/direct-export`
- Direct staging artifacts: `/home/asus/hunyuanocr-recovery/phase4j/direct-staging-artifacts`

## Remaining Risks

- The PyTorch reference capture is now automated for the current smoke and
  regression image set, but it still assumes the local HF model directory and
  reference Python environment are already provisioned.
- Windows validation was not rerun in Phase 4J. The regenerated artifacts are
  byte-identical to the verified artifacts and Linux validation passed, but a
  later release pass should rerun the Windows package matrix after the
  reference-capture pipeline changes.
- The current reference capture pipeline is optimized for correctness and
  auditability. A future pass can reduce model reloads by merging compatible
  capture steps into fewer PyTorch processes.
