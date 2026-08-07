from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]

SMOKE_CASE = {
    "name": "ocr_smoke_en",
    "path": "tests/assets/ocr_smoke_en.png",
    "grid_thw": [1, 22, 50],
    "sequence_length": 313,
    "image_token_span": [2, 290],
    "generated_token_ids": [
        93892,
        5112,
        206,
        1717,
        21,
        185,
        18009,
        15613,
        16678,
        21836,
        120007,
    ],
    "generated_text": "HELLO 2026\nNCNN CPU TEST",
    "eos_reached": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run release-grade HunyuanOCR runtime validation suites."
    )
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=(
            "smoke",
            "dynamic",
            "real-png",
            "real-jpeg",
            "cache-budgets",
            "error-paths",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--packing", choices=("0", "1", "both"), default="both")
    parser.add_argument("--verify", choices=("none", "size", "sha256"), default="size")
    parser.add_argument("--threads", type=int, default=9)
    return parser.parse_args()


def match_required(pattern: str, text: str, flags: int = re.MULTILINE) -> re.Match[str]:
    match = re.search(pattern, text, flags)
    if match is None:
        raise RuntimeError(f"Missing CLI output pattern: {pattern}")
    return match


def parse_output(output: str) -> dict[str, object]:
    token_text = match_required(r"^Generated tokens:(.*)$", output).group(1).strip()
    generated_text = match_required(
        r"^Generated text:\n(.*?)\n\nLoad seconds",
        output,
        re.MULTILINE | re.DOTALL,
    ).group(1)
    return {
        "grid_thw": [
            int(value)
            for value in match_required(r"^Image grid\s+: \[([^]]+)\]$", output)
            .group(1)
            .split(",")
        ],
        "image_token_span": [
            int(value)
            for value in match_required(
                r"^Image token span: \[([0-9]+),([0-9]+)\)$", output
            ).groups()
        ],
        "sequence_length": int(
            match_required(r"^Prefill length\s+: ([0-9]+)$", output).group(1)
        ),
        "generated_token_ids": [int(value) for value in token_text.split()],
        "generated_text": generated_text,
        "eos_reached": (
            match_required(r"^EOS reached\s+: (true|false)$", output).group(1)
            == "true"
        ),
        "runtime_seconds": float(
            match_required(r"^Runtime seconds\s+: ([0-9.]+)$", output).group(1)
        ),
        "decode_seconds": float(
            match_required(r"^Decode seconds\s+: ([0-9.]+)$", output).group(1)
        ),
        "peak_rss_kib": int(
            match_required(r"^Peak RSS KiB\s+: ([0-9]+)$", output).group(1)
        ),
        "memory_cached_decoder_layers": int(
            match_required(r"^Memory layers\s+: ([0-9]+)$", output).group(1)
        ),
        "file_streamed_decoder_layers": int(
            match_required(r"^File layers\s+: ([0-9]+)$", output).group(1)
        ),
        "decoder_cache_estimated_mib": int(
            match_required(r"^Cache estimate\s+: ([0-9]+) MiB$", output).group(1)
        ),
    }


def packing_modes(value: str) -> list[int]:
    return [0, 1] if value == "both" else [int(value)]


def load_cases(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def real_case(name: str) -> dict[str, object]:
    cases = load_cases(PROJECT_DIR / "tests/assets/real_ocr_expected.json")
    return next(case for case in cases if case["name"] == name)


def run_cli_case(
    args: argparse.Namespace,
    case: dict[str, object],
    packing: int,
    max_tokens: int,
    decoder_cache_mib: int,
    tag: str,
) -> dict[str, object]:
    log_path = args.log_dir / f"{tag}_packing{packing}_cache{decoder_cache_mib}.log"
    command = [
        str(args.cli),
        "--model-dir",
        str(args.model_dir),
        "--image",
        str(PROJECT_DIR / str(case["path"])),
        "--packing",
        str(packing),
        "--threads",
        str(args.threads),
        "--max-new-tokens",
        str(max_tokens),
        "--decoder-cache-mib",
        str(decoder_cache_mib),
        "--verify",
        args.verify,
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{tag} packing={packing} failed; see {log_path}")
    actual = parse_output(completed.stdout)
    for key in (
        "grid_thw",
        "image_token_span",
        "sequence_length",
        "generated_token_ids",
        "generated_text",
        "eos_reached",
    ):
        if actual[key] != case[key]:
            raise RuntimeError(
                f"{tag} packing={packing} {key}: "
                f"{actual[key]!r} != {case[key]!r}"
            )
    result = {
        "suite_case": tag,
        "case": case["name"],
        "packing": packing,
        "decoder_cache_mib": decoder_cache_mib,
        "wall_seconds": time.perf_counter() - start,
        "log": str(log_path),
        **actual,
    }
    print(
        f"PASS {tag} packing={packing} cache={decoder_cache_mib} "
        f"tokens={len(actual['generated_token_ids'])}"
    )
    return result


def run_exact_suite(
    args: argparse.Namespace,
    suite: str,
) -> list[dict[str, object]]:
    if suite == "smoke":
        cases = [SMOKE_CASE]
        max_tokens = 32
    elif suite == "dynamic":
        cases = load_cases(PROJECT_DIR / "tests/assets/dynamic_ocr_expected.json")
        max_tokens = 32
    elif suite == "real-png":
        cases = [real_case("ocr_receipt_real_png")]
        max_tokens = 256
    elif suite == "real-jpeg":
        cases = [real_case("ocr_receipt_real_jpeg")]
        max_tokens = 256
    else:
        raise RuntimeError(f"Unsupported exact suite: {suite}")
    results = []
    for case in cases:
        for packing in packing_modes(args.packing):
            results.append(
                run_cli_case(
                    args,
                    case,
                    packing,
                    max_tokens,
                    512,
                    f"{suite}_{case['name']}",
                )
            )
    return results


def run_cache_budget_suite(args: argparse.Namespace) -> list[dict[str, object]]:
    results = []
    for budget in (0, 512, 2048):
        for packing in packing_modes(args.packing):
            result = run_cli_case(
                args,
                SMOKE_CASE,
                packing,
                32,
                budget,
                "cache_budget_smoke",
            )
            memory_layers = int(result["memory_cached_decoder_layers"])
            if budget == 0 and memory_layers != 0:
                raise RuntimeError("0 MiB cache budget unexpectedly cached layers")
            if budget == 512 and memory_layers <= 0:
                raise RuntimeError("512 MiB cache budget did not cache any layers")
            if budget == 2048 and memory_layers != 24:
                raise RuntimeError("2048 MiB cache budget did not cache all 24 layers")
            results.append(result)
    return results


def run_expected_failure(
    command: list[str],
    log_path: Path,
    required_pattern: str,
) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode == 0:
        raise RuntimeError(f"Expected failure succeeded unexpectedly: {log_path}")
    if re.search(required_pattern, completed.stdout) is None:
        raise RuntimeError(
            f"Expected failure did not contain {required_pattern!r}: {log_path}"
        )
    print(f"PASS expected failure {log_path.stem}")
    return {
        "case": log_path.stem,
        "status": "passed",
        "log": str(log_path),
        "pattern": required_pattern,
    }


def prepare_synthetic_model(root: Path, compatibility_min_version: str) -> Path:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "runtime_manifest.tsv").write_text(
        "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1\nfile_count\t0\n",
        encoding="ascii",
    )
    (root / "runtime_compatibility.tsv").write_text(
        "\n".join(
            [
                "HUNYUANOCR_NCNN_RUNTIME_COMPATIBILITY_V1",
                "model_id\ttencent/HunyuanOCR",
                "runtime_abi_major\t0",
                f"runtime_min_version\t{compatibility_min_version}",
                "runtime_max_exclusive_version\t100.0.0",
                "manifest_format\tHUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1",
                "file_count\t0",
                "precision\tfp32",
                "jpeg_pixel_contract\tstb_rgb_v1",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    return root


def prepare_missing_file_model(root: Path) -> Path:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "runtime_manifest.tsv").write_text(
        "\n".join(
            [
                "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1",
                "file_count\t1",
                "missing/missing.ncnn.param\t1\t"
                "0000000000000000000000000000000000000000000000000000000000000000",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    return root


def run_error_path_suite(args: argparse.Namespace) -> list[dict[str, object]]:
    results = []
    results.append(
        run_expected_failure(
            [
                str(args.cli),
                "--model-dir",
                str(args.model_dir),
                "--image",
                str(PROJECT_DIR / SMOKE_CASE["path"]),
                "--cache-decode",
                "1",
                "--decoder-cache-mib",
                "512",
            ],
            args.log_dir / "error_cache_mode_conflict.log",
            r"mutually exclusive",
        )
    )
    results.append(
        run_expected_failure(
            [
                str(args.cli),
                "--model-dir",
                str(prepare_synthetic_model(args.log_dir / "compat_too_new", "99.0.0")),
                "--image",
                str(PROJECT_DIR / SMOKE_CASE["path"]),
            ],
            args.log_dir / "error_runtime_too_old.log",
            r"older than model minimum runtime",
        )
    )
    results.append(
        run_expected_failure(
            [
                str(args.cli),
                "--model-dir",
                str(prepare_missing_file_model(args.log_dir / "missing_file_model")),
                "--image",
                str(PROJECT_DIR / SMOKE_CASE["path"]),
            ],
            args.log_dir / "error_manifest_missing_file.log",
            r"Missing model file",
        )
    )
    results.append(
        run_expected_failure(
            [
                str(args.cli),
                "--model-dir",
                str(args.model_dir),
                "--image",
                str(PROJECT_DIR / SMOKE_CASE["path"]),
                "--max-vision-patches",
                "1",
                "--verify",
                args.verify,
            ],
            args.log_dir / "error_patch_limit.log",
            r"exceeding the configured limit",
        )
    )
    return results


def write_summary(args: argparse.Namespace, results: list[dict[str, object]]) -> None:
    summary_path = args.log_dir / f"{args.suite}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "status": "passed",
                "result_count": len(results),
                "results": results,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    suites = (
        ["smoke", "dynamic", "real-png", "real-jpeg", "cache-budgets", "error-paths"]
        if args.suite == "all"
        else [args.suite]
    )
    results: list[dict[str, object]] = []
    for suite in suites:
        if suite in ("smoke", "dynamic", "real-png", "real-jpeg"):
            results.extend(run_exact_suite(args, suite))
        elif suite == "cache-budgets":
            results.extend(run_cache_budget_suite(args))
        elif suite == "error-paths":
            results.extend(run_error_path_suite(args))
        else:
            raise RuntimeError(f"Unsupported suite: {suite}")
    write_summary(args, results)


if __name__ == "__main__":
    main()
