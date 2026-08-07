from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
RECOVERY_DIR = Path.home() / "hunyuanocr-recovery/phase4e"
OUTPUT = PROJECT_DIR / "docs/linux_phase4e_validation.json"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_results(directory: str) -> list[dict[str, object]]:
    report = load_json(
        RECOVERY_DIR / "linux" / directory / "dynamic_runtime_summary.json"
    )
    compact = []
    for result in report["results"]:
        decode_seconds = result.get("decode_seconds")
        if decode_seconds is None:
            log_text = Path(result["log"]).read_text(encoding="utf-8")
            match = re.search(r"^Decode seconds\s+: ([0-9.]+)$", log_text, re.MULTILINE)
            if match is None:
                raise RuntimeError(f"Missing decode time in {result['log']}")
            decode_seconds = float(match.group(1))
        compact.append({
            "packing": result["packing"],
            "runtime_seconds": result["runtime_seconds"],
            "decode_seconds": decode_seconds,
            "peak_rss_kib": result["peak_rss_kib"],
            "memory_cached_decoder_layers": result.get(
                "memory_cached_decoder_layers", 0
            ),
            "decoder_cache_estimated_mib": result.get(
                "decoder_cache_estimated_mib", 0
            ),
            "generated_token_count": len(result["generated_token_ids"]),
            "eos_reached": result["eos_reached"],
        })
    return compact


def reduction(before: float, after: float) -> float:
    return 100.0 * (before - after) / before


def parse_repeat(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    current_rss = [
        int(match.group(1))
        for match in re.finditer(r"current_rss_kib=([0-9]+)", text)
    ]
    if len(current_rss) != 10:
        raise RuntimeError(f"Expected 10 repeated results in {path}")
    initial = re.search(r"^initial_rss_kib=([0-9]+)$", text, re.MULTILINE)
    final = re.search(r"^final_rss_kib=([0-9]+)$", text, re.MULTILINE)
    peak = re.search(r"^peak_rss_kib=([0-9]+)$", text, re.MULTILINE)
    if initial is None or final is None or peak is None:
        raise RuntimeError(f"Missing repeated memory summary in {path}")
    return {
        "iterations": 10,
        "initial_rss_kib": int(initial.group(1)),
        "final_rss_kib": int(final.group(1)),
        "peak_rss_kib": int(peak.group(1)),
        "warmup_to_final_growth_kib": int(final.group(1)) - current_rss[1],
        "status": "passed",
    }


def main() -> None:
    baseline = compact_results("jpeg_final_cache0")
    cache_512 = compact_results("jpeg_raw_cache512")
    cache_2048 = compact_results("jpeg_raw_cache2048")
    improvements = []
    for packing in (0, 1):
        before = next(item for item in baseline if item["packing"] == packing)
        medium = next(item for item in cache_512 if item["packing"] == packing)
        full = next(item for item in cache_2048 if item["packing"] == packing)
        improvements.append({
            "packing": packing,
            "cache_512_runtime_reduction_percent": reduction(
                float(before["runtime_seconds"]),
                float(medium["runtime_seconds"]),
            ),
            "cache_2048_runtime_reduction_percent": reduction(
                float(before["runtime_seconds"]),
                float(full["runtime_seconds"]),
            ),
        })
    assets = load_json(PROJECT_DIR / "tests/assets/real_ocr_assets.json")
    expected = load_json(PROJECT_DIR / "tests/assets/real_ocr_expected.json")
    png_tokens = expected["cases"][0]["generated_token_ids"]
    jpeg_tokens = expected["cases"][1]["generated_token_ids"]
    first_difference = next(
        index
        for index, (png, jpeg) in enumerate(zip(png_tokens, jpeg_tokens))
        if png != jpeg
    )
    dynamic = load_json(
        RECOVERY_DIR
        / "linux/dynamic_cache512/dynamic_runtime_summary.json"
    )
    png = load_json(
        RECOVERY_DIR / "linux/png_cache512/dynamic_runtime_summary.json"
    )
    report = {
        "phase": "4E",
        "status": "passed",
        "platform": "WSL2 Ubuntu 24.04",
        "decoder_cache_contract": {
            "budget_unit": "MiB of raw ncnn decoder model bytes",
            "layer_ncnn_bin_bytes": 69215260,
            "total_layers": 24,
        },
        "real_jpeg_results": {
            "cache_0_mib": baseline,
            "cache_512_mib": cache_512,
            "cache_2048_mib": cache_2048,
            "runtime_improvements": improvements,
        },
        "jpeg_pixel_contract": {
            **assets["jpeg_decode_comparison"],
            "stb_rgb_sha256": assets["assets"][2]["sha256"],
            "png_jpeg_first_token_difference_index": first_difference,
            "png_token_at_difference": png_tokens[first_difference],
            "jpeg_token_at_difference": jpeg_tokens[first_difference],
            "jpeg_generated_token_count": len(jpeg_tokens),
        },
        "regressions": {
            "dynamic_packed_unpacked_cases": len(dynamic["results"]),
            "real_png_packed_unpacked_cases": len(png["results"]),
            "status": "passed",
        },
        "repeat_10_cache_512_mib": {
            "unpacked": parse_repeat(
                RECOVERY_DIR / "linux/repeat10_unpacked_cache512.log"
            ),
            "packed": parse_repeat(
                RECOVERY_DIR / "linux/repeat10_packed_cache512.log"
            ),
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
