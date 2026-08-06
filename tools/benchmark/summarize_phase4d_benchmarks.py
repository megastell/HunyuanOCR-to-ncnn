from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
RECOVERY_DIR = Path.home() / "hunyuanocr-recovery/phase4d"
OUTPUT = PROJECT_DIR / "docs/linux_phase4d_validation.json"


def parse_repeat(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    iterations = []
    for line in text.splitlines():
        if not line.startswith("iteration="):
            continue
        values = {
            key: value
            for key, value in re.findall(r"([a-z_]+)=([^ ]+)", line)
        }
        iterations.append({
            "iteration": int(values["iteration"]),
            "grid": [int(value) for value in values["grid"].split(",")],
            "runtime_seconds": float(values["runtime_seconds"]),
            "current_rss_kib": int(values["current_rss_kib"]),
            "peak_rss_kib": int(values["peak_rss_kib"]),
        })
    if len(iterations) != 10:
        raise RuntimeError(f"Expected 10 iterations in {path}")
    final = re.search(r"^final_rss_kib=([0-9]+)$", text, re.MULTILINE)
    peaks = re.findall(r"^peak_rss_kib=([0-9]+)$", text, re.MULTILINE)
    if final is None or not peaks:
        raise RuntimeError(f"Missing memory summary in {path}")
    return {
        "iterations": iterations,
        "final_rss_kib": int(final.group(1)),
        "peak_rss_kib": int(peaks[-1]),
        "warmup_to_final_growth_kib": (
            int(final.group(1)) - iterations[1]["current_rss_kib"]
        ),
        "warm_runtime_seconds_mean": sum(
            item["runtime_seconds"] for item in iterations[4:]
        ) / 6,
    }


def reduction(before: int, after: int) -> float:
    return 100.0 * (before - after) / before


def main() -> None:
    modes = {}
    for name in ("unpacked", "packed"):
        before = parse_repeat(
            RECOVERY_DIR / "baseline" / f"linux_{name}_repeat10.log"
        )
        after = parse_repeat(
            RECOVERY_DIR / "optimized" / f"linux_{name}_repeat10.log"
        )
        modes[name] = {
            "before": before,
            "after": after,
            "peak_rss_reduction_percent": reduction(
                before["peak_rss_kib"], after["peak_rss_kib"]
            ),
            "final_rss_reduction_percent": reduction(
                before["final_rss_kib"], after["final_rss_kib"]
            ),
        }
    dynamic = json.loads((
        RECOVERY_DIR / "linux_dynamic/dynamic_runtime_summary.json"
    ).read_text(encoding="utf-8"))
    real = json.loads((
        RECOVERY_DIR / "linux_real/dynamic_runtime_summary.json"
    ).read_text(encoding="utf-8"))
    report = {
        "phase": "4D",
        "status": "passed",
        "platform": "WSL2 Ubuntu 24.04",
        "repeat_modes": modes,
        "dynamic_exact_results": dynamic["results"],
        "real_ocr_exact_results": real["results"],
        "jpeg_compatibility": {
            "status": "passed",
            "pyTorch_exact_parity": (
                "not claimed; stb and Pillow JPEG pixels differ"
            ),
        },
        "patch_limit": {
            "status": "passed",
            "default_max_vision_patches": 2048,
            "rejected_grid": [1, 120, 56],
            "rejected_patch_count": 6720,
        },
        "persistent_log_root": str(RECOVERY_DIR),
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
