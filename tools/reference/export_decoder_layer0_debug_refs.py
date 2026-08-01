from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

SOURCE_DIR = (
    PROJECT_DIR
    / "reference/smoke_en_cpu_fp32"
    / "decoder_layer0_prefill"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "artifacts/decoder_layer0_prefill"
    / "debug_refs"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/decoder_layer0_debug_refs.json"
)

NAMES = (
    "input_layernorm_output",
    "q_projection_output",
    "k_projection_output",
    "v_projection_output",
    "query_layernorm_output",
    "key_layernorm_output",
    "attention_output",
    "o_projection_output",
    "post_attention_layernorm_output",
    "mlp_gate_output",
    "mlp_up_output",
    "mlp_down_output",
    "mlp_output",
    "layer_output",
)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report: dict[str, object] = {}

    for name in NAMES:
        source = SOURCE_DIR / f"{name}.npy"

        if not source.is_file():
            raise FileNotFoundError(source)

        value = np.load(source)

        if value.dtype != np.float32:
            value = value.astype(
                np.float32,
                copy=False,
            )

        value = np.ascontiguousarray(value)

        destination = (
            OUTPUT_DIR / f"{name}_f32.bin"
        )

        value.tofile(destination)

        report[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.size),
            "bytes": destination.stat().st_size,
            "path": destination.relative_to(PROJECT_DIR).as_posix(),
        }

        print(
            f"{name:36s} "
            f"shape={value.shape!s:24s} "
            f"bytes={destination.stat().st_size}"
        )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("Saved:", REPORT_PATH)


if __name__ == "__main__":
    main()
