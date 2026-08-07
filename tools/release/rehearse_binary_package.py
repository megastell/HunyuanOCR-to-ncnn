from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
SMOKE_TOKENS = [93892, 5112, 206, 1717, 21, 185, 18009, 15613, 16678, 21836, 120007]
SMOKE_TEXT = "HELLO 2026\nNCNN CPU TEST"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rehearse a binary package from clean extraction to OCR."
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_DIR / "tests/assets/ocr_smoke_en.png",
    )
    parser.add_argument("--packing", choices=("0", "1"), default="0")
    parser.add_argument("--threads", type=int, default=9)
    parser.add_argument("--decoder-cache-mib", type=int, default=512)
    return parser.parse_args()


def extract_package(package: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    if package.suffix == ".zip":
        with zipfile.ZipFile(package) as archive:
            archive.extractall(destination)
    elif package.name.endswith(".tar.gz"):
        with tarfile.open(package, "r:gz") as archive:
            archive.extractall(destination)
    else:
        raise RuntimeError(f"Unsupported package format: {package}")
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one extracted package root in {destination}")
    return roots[0]


def match_required(pattern: str, text: str, flags: int = re.MULTILINE) -> re.Match[str]:
    match = re.search(pattern, text, flags)
    if match is None:
        raise RuntimeError(f"Missing CLI output pattern: {pattern}")
    return match


def validate_output(text: str) -> dict[str, object]:
    tokens = [
        int(value)
        for value in match_required(r"^Generated tokens:(.*)$", text)
        .group(1)
        .strip()
        .split()
    ]
    generated_text = match_required(
        r"^Generated text:\n(.*?)\n\nLoad seconds",
        text,
        re.MULTILINE | re.DOTALL,
    ).group(1)
    if tokens != SMOKE_TOKENS:
        raise RuntimeError(f"Unexpected generated tokens: {tokens}")
    if generated_text != SMOKE_TEXT:
        raise RuntimeError(f"Unexpected generated text: {generated_text!r}")
    if match_required(r"^EOS reached\s+: (true|false)$", text).group(1) != "true":
        raise RuntimeError("EOS was not reached")
    return {
        "generated_token_ids": tokens,
        "generated_text": generated_text,
        "runtime_seconds": float(
            match_required(r"^Runtime seconds\s+: ([0-9.]+)$", text).group(1)
        ),
        "peak_rss_kib": int(
            match_required(r"^Peak RSS KiB\s+: ([0-9]+)$", text).group(1)
        ),
    }


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    package_root = extract_package(args.package, args.work_dir / "extract")
    required_docs = [
        package_root / "share/doc/HunyuanOCR_ncnn/LICENSE",
        package_root / "share/doc/HunyuanOCR_ncnn/NOTICE",
        package_root / "share/doc/HunyuanOCR_ncnn/THIRD_PARTY_NOTICES.md",
        package_root
        / "share/doc/HunyuanOCR_ncnn/third_party/licenses/ncnn-LICENSE.txt",
        package_root
        / "share/doc/HunyuanOCR_ncnn/third_party/licenses/Tencent-HunyuanOCR-LICENSE.txt",
        package_root
        / "share/doc/HunyuanOCR_ncnn/third_party/licenses/stb_image-LICENSE.txt",
    ]
    missing = [str(path) for path in required_docs if not path.is_file()]
    if missing:
        raise RuntimeError(f"Package is missing release notice files: {missing}")
    input_dir = args.work_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    image = input_dir / args.image.name
    shutil.copy2(args.image, image)
    cli = package_root / "bin/hunyuanocr_cli"
    if not cli.is_file():
        raise RuntimeError(f"Extracted CLI not found: {cli}")
    log_path = args.log_dir / f"package_rehearsal_packing{args.packing}.log"
    command = [
        str(cli),
        "--model-dir",
        str(args.model_dir),
        "--image",
        str(image),
        "--packing",
        args.packing,
        "--threads",
        str(args.threads),
        "--max-new-tokens",
        "32",
        "--decoder-cache-mib",
        str(args.decoder_cache_mib),
        "--verify",
        "size",
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Extracted CLI failed; see {log_path}")
    result = validate_output(completed.stdout)
    report = {
        "phase": "4G",
        "status": "passed",
        "package": str(args.package),
        "package_root": str(package_root),
        "model_directory": str(args.model_dir),
        "input_image": str(image),
        "packing": int(args.packing),
        "decoder_cache_mib": args.decoder_cache_mib,
        "wall_seconds": time.perf_counter() - start,
        "notice_files_checked": [str(path) for path in required_docs],
        "ocr": result,
        "log": str(log_path),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(args.report)


if __name__ == "__main__":
    main()
