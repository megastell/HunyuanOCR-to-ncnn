# Phase 4K Reproducible Release Acceptance

## Status

Phase 4K is complete on Linux.

The project now exposes a formal reproducible artifact acceptance path that
runs from a clean persistent staging directory:

1. Regenerate PyTorch reference tensors from the local HuggingFace model,
   original OCR images, and fixed OCR prompt.
2. Run the direct pnnx/ncnn exporters into staging.
3. Generate `runtime_manifest.tsv` and `runtime_compatibility.tsv`.
4. Run Linux smoke, dynamic-size, real PNG, real JPEG, and cache-budget
   validation against the staged runtime artifacts.
5. Verify that the existing repository `artifacts/` and `reference/` trees were
   not modified.

## Entry Points

- Script:
  `tools/release/reproduce_runtime_artifacts_acceptance.py`
- Persistent shell wrapper:
  `tools/release/run_phase4k_repro_acceptance.sh`
- CTest:
  `hunyuanocr_reproduce_runtime_artifacts_acceptance`
- CMake option:
  `-DHUNYUANOCR_ENABLE_REPRODUCTION_TESTS=ON`

## Acceptance Run

Command:

```bash
bash tools/release/run_phase4k_repro_acceptance.sh
```

Result:

- CTest status: passed
- Test count: 1
- CTest wall time: 1216.66 seconds
- Acceptance report:
  `docs/phase4k_reproducible_release_acceptance.json`
- Persistent staging root:
  `/home/asus/hunyuanocr-recovery/phase4k`

## Reproduction Evidence

- Reference capture steps: 7
- Regenerated reference `.npy` files: 1255
- Direct export elapsed time: 1103.4 seconds
- Runtime artifact file count: 170
- Runtime artifact total size: 5.724460432305932 GiB
- Staging artifact bytes: 24349436516
- Total Phase 4K work directory bytes: 26036477975
- Manifest structure comparison: passed
- Manifest changed digest/size count vs verified artifacts: 0
- Existing `artifacts/` and `reference/` unchanged: true

## Linux Validation Suites

- smoke: passed in 32.4 seconds
- dynamic: passed in 66.6 seconds
- real-png: passed in 242.2 seconds
- real-jpeg: passed in 250.6 seconds
- cache-budgets: passed in 80.4 seconds

## Key SHA-256 Values

- Staging `runtime_manifest.tsv`:
  `71498acaeafff31e2cbfa4c3ed9de81b73d9078e1f2bc7528e87bc36d7222431`
- Staging `runtime_compatibility.tsv`:
  `cc47674acdbd3770952294b9363952fbea347acbeb355a7d363bd7e6c86c73f6`

## Dependency Environment

- Python reference environment:
  `/home/asus/work/hunyuanocr/.venv-reference/bin/python`
- PyTorch: 2.12.1+cpu
- Transformers: 5.13.0
- NumPy: 2.5.1
- Pillow: 12.3.0
- pnnx:
  `/home/asus/work/hunyuanocr/.venv-pnnx/bin/pnnx`

## Remaining Risks

- The full reproducible acceptance is intentionally slow and large; it is
  registered as an explicit opt-in CTest rather than a default build test.
- The pipeline depends on the local HF model directory and local pnnx/reference
  Python environments being present and version-compatible.
- Windows consumes artifacts produced by this Linux reproduction pipeline; a
  matching Windows-side copy and validation pass is still required before a
  final dual-platform release artifact refresh.
