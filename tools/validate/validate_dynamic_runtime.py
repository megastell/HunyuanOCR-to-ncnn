from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dynamic-grid OCR output against PyTorch references."
    )
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--expected",
        type=Path,
        default=PROJECT_DIR / "tests/assets/dynamic_ocr_expected.json",
    )
    parser.add_argument("--packing", choices=("0", "1", "both"), default="both")
    parser.add_argument("--verify", choices=("none", "size", "sha256"), default="size")
    parser.add_argument("--threads", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def parse_output(output: str) -> dict[str, object]:
    def match(pattern: str, flags: int = re.MULTILINE) -> re.Match[str]:
        result = re.search(pattern, output, flags)
        if result is None:
            raise RuntimeError(f"Missing CLI output pattern: {pattern}")
        return result

    token_text = match(r"^Generated tokens:(.*)$").group(1).strip()
    text = match(
        r"^Generated text:\n(.*?)\n\nLoad seconds",
        re.MULTILINE | re.DOTALL,
    ).group(1)
    return {
        "grid_thw": [
            int(value)
            for value in match(r"^Image grid\s+: \[([^]]+)\]$")
            .group(1).split(",")
        ],
        "image_token_span": [
            int(value)
            for value in match(r"^Image token span: \[([0-9]+),([0-9]+)\)$").groups()
        ],
        "sequence_length": int(match(r"^Prefill length\s+: ([0-9]+)$").group(1)),
        "generated_token_ids": [int(value) for value in token_text.split()],
        "generated_text": text,
        "eos_reached": match(r"^EOS reached\s+: (true|false)$").group(1) == "true",
        "peak_rss_kib": int(match(r"^Peak RSS KiB\s+: ([0-9]+)$").group(1)),
        "runtime_seconds": float(match(r"^Runtime seconds\s+: ([0-9.]+)$").group(1)),
        "load_seconds": float(match(r"^Load seconds\s+: ([0-9.]+)$").group(1)),
    }


def main() -> None:
    args = parse_args()
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    modes = [0, 1] if args.packing == "both" else [int(args.packing)]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for case in expected["cases"]:
        for packing in modes:
            command = [
                str(args.cli),
                "--model-dir", str(args.model_dir),
                "--image", str(PROJECT_DIR / case["path"]),
                "--packing", str(packing),
                "--threads", str(args.threads),
                "--max-new-tokens", str(args.max_new_tokens),
                "--verify", args.verify,
            ]
            start = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_path = args.log_dir / f"{case['name']}_packing{packing}.log"
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{case['name']} packing={packing} failed; see {log_path}"
                )
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
                        f"{case['name']} packing={packing} {key}: "
                        f"{actual[key]!r} != {case[key]!r}"
                    )
            result = {
                "case": case["name"],
                "packing": packing,
                "wall_seconds": time.perf_counter() - start,
                **actual,
                "log": str(log_path),
            }
            results.append(result)
            print(
                f"PASS {case['name']} packing={packing} "
                f"grid={actual['grid_thw']} tokens={actual['generated_token_ids']}"
            )
    summary_path = args.log_dir / "dynamic_runtime_summary.json"
    summary_path.write_text(
        json.dumps({"results": results}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
