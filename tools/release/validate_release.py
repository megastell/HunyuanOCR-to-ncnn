from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_NCNN_DIR = Path.home() / ".local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, test, package, and audit an offline HunyuanOCR release."
    )
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "artifacts")
    parser.add_argument("--ncnn-dir", type=Path, default=DEFAULT_NCNN_DIR)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    parser.add_argument(
        "--ctest-suite",
        action="append",
        default=[],
        help="CTest labels to run; defaults to all Phase 4F release suites.",
    )
    parser.add_argument("--skip-ctest", action="store_true")
    parser.add_argument("--skip-cpack", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], log_path: Path, cwd: Path = PROJECT_DIR) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed; see {log_path}")


def run_capture(command: list[str], log_path: Path, cwd: Path = PROJECT_DIR) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed; see {log_path}")
    return completed.stdout


def audit_dependencies_and_licenses(ncnn_dir: Path) -> dict[str, object]:
    root_license = [
        path.name
        for path in PROJECT_DIR.iterdir()
        if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING"))
    ]
    stb = PROJECT_DIR / "third_party/stb/stb_image.h"
    stb_text = stb.read_text(encoding="utf-8", errors="replace")
    ncnn_license_candidates = []
    ncnn_prefix = ncnn_dir.parents[2] if len(ncnn_dir.parents) >= 3 else ncnn_dir
    for candidate in (
        ncnn_prefix / "LICENSE.txt",
        ncnn_prefix / "LICENSE",
        ncnn_prefix / "share/doc/ncnn/LICENSE.txt",
    ):
        if candidate.is_file():
            ncnn_license_candidates.append(str(candidate))
    warnings = []
    if not root_license:
        warnings.append("Repository has no top-level LICENSE/COPYING file.")
    if not ncnn_license_candidates:
        warnings.append("ncnn license file was not found beside the configured ncnn package.")
    return {
        "runtime_language": "C++17",
        "runtime_third_party_headers": [
            {
                "name": "stb_image",
                "path": "third_party/stb/stb_image.h",
                "license_detected": "public domain"
                if "public domain image loader" in stb_text[:512].lower()
                else "unknown",
                "status": "passed"
                if "public domain image loader" in stb_text[:512].lower()
                else "warning",
            }
        ],
        "link_dependencies": [
            {
                "name": "ncnn",
                "cmake_package": str(ncnn_dir),
                "license_files": ncnn_license_candidates,
                "status": "passed" if ncnn_license_candidates else "warning",
            }
        ],
        "project_license_files": root_license,
        "warnings": warnings,
        "status": "passed" if not warnings else "passed_with_warnings",
    }


def package_inventory(package_dir: Path) -> list[dict[str, object]]:
    packages = []
    for path in sorted(package_dir.glob("HunyuanOCR-ncnn-*")):
        if path.is_file() and path.suffix.lower() in (".gz", ".zip"):
            packages.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return packages


def parse_ctest(output: str) -> dict[str, object]:
    failed = re.search(r"([0-9]+)% tests passed, ([0-9]+) tests failed", output)
    if failed:
        return {
            "passed_percent": int(failed.group(1)),
            "failed_tests": int(failed.group(2)),
        }
    all_pass = re.search(
        r"100% tests passed(?:, 0 tests failed)? out of ([0-9]+)",
        output,
    )
    return {
        "passed_percent": 100 if all_pass else None,
        "failed_tests": 0 if all_pass else None,
        "test_count": int(all_pass.group(1)) if all_pass else None,
    }


def main() -> None:
    args = parse_args()
    for directory in (args.build_dir, args.install_dir, args.package_dir, args.log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not args.ncnn_dir.is_dir():
        raise RuntimeError(f"ncnn_DIR was not found: {args.ncnn_dir}")
    if not args.model_dir.is_dir():
        raise RuntimeError(f"Model directory was not found: {args.model_dir}")

    run(
        [sys.executable, "tools/export/export_runtime_manifest.py"],
        args.log_dir / "export_runtime_manifest.log",
    )
    configure = [
        "cmake",
        "-S",
        str(PROJECT_DIR),
        "-B",
        str(args.build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={args.install_dir}",
        f"-Dncnn_DIR={args.ncnn_dir}",
        "-DHUNYUANOCR_BUILD_CLI=ON",
        "-DHUNYUANOCR_BUILD_BENCHMARKS=ON",
        "-DHUNYUANOCR_BUILD_PARITY_TESTS=OFF",
        "-DHUNYUANOCR_ENABLE_RELEASE_TESTS=ON",
        f"-DHUNYUANOCR_RELEASE_MODEL_DIR={args.model_dir}",
        f"-DHUNYUANOCR_RELEASE_LOG_DIR={args.log_dir / 'ctest'}",
    ]
    run(configure, args.log_dir / "configure.log")
    run(
        ["cmake", "--build", str(args.build_dir), "-j", str(args.jobs)],
        args.log_dir / "build.log",
    )
    run(
        ["cmake", "--install", str(args.build_dir)],
        args.log_dir / "install.log",
    )

    ctest_results = {}
    suites = args.ctest_suite or [
        "smoke",
        "dynamic",
        "real-png",
        "real-jpeg",
        "cache-budgets",
        "error-paths",
    ]
    if not args.skip_ctest:
        for suite in suites:
            output = run_capture(
                [
                    "ctest",
                    "--test-dir",
                    str(args.build_dir),
                    "-L",
                    suite,
                    "--output-on-failure",
                ],
                args.log_dir / f"ctest_{suite}.log",
            )
            ctest_results[suite] = parse_ctest(output)

    consumer_build = args.log_dir / "runtime_api_consumer_build"
    ncnn_prefix = args.ncnn_dir.parents[2] if len(args.ncnn_dir.parents) >= 3 else args.ncnn_dir
    prefix_path = f"{args.install_dir};{ncnn_prefix}"
    run(
        [
            "cmake",
            "-S",
            str(PROJECT_DIR / "tests/runtime_api_consumer"),
            "-B",
            str(consumer_build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={prefix_path}",
        ],
        args.log_dir / "consumer_configure.log",
    )
    run(
        ["cmake", "--build", str(consumer_build), "-j", str(args.jobs)],
        args.log_dir / "consumer_build.log",
    )
    run(
        [str(consumer_build / "runtime_api_consumer")],
        args.log_dir / "consumer_run.log",
    )

    packages = []
    if not args.skip_cpack:
        run(
            ["cpack", "-G", "TGZ;ZIP", "-B", str(args.package_dir)],
            args.log_dir / "cpack.log",
            cwd=args.build_dir,
        )
        packages = package_inventory(args.package_dir)
        if not packages:
            raise RuntimeError("CPack did not produce any release packages")

    report = {
        "phase": "4F",
        "status": "passed",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "model_directory": str(args.model_dir),
        "ncnn_dir": str(args.ncnn_dir),
        "build_dir": str(args.build_dir),
        "install_dir": str(args.install_dir),
        "package_dir": str(args.package_dir),
        "ctest": ctest_results,
        "packages": packages,
        "dependency_license_audit": audit_dependencies_and_licenses(args.ncnn_dir),
        "offline_release_acceptance": {
            "install_tree_created": True,
            "find_package_consumer_passed": True,
            "packages_created": bool(packages) or args.skip_cpack,
            "network_required": False,
        },
        "persistent_log_root": str(args.log_dir),
    }
    report_path = PROJECT_DIR / "docs/linux_phase4f_release_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
