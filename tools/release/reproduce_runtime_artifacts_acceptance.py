from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = Path.home() / "hunyuanocr-recovery" / "phase4k"
DEFAULT_HF_MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
DEFAULT_REFERENCE_PYTHON = (
    Path.home() / "work/hunyuanocr/.venv-reference/bin/python"
)
DEFAULT_PNNX = Path.home() / "work/hunyuanocr/.venv-pnnx/bin/pnnx"
DEFAULT_CLI = (
    Path.home() / "hunyuanocr-recovery/phase4g/linux-release-build/hunyuanocr_cli"
)
DEFAULT_REPORT = PROJECT_DIR / "docs/phase4k_reproducible_release_acceptance.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible release acceptance path: regenerate PyTorch "
            "references, directly export pnnx/ncnn runtime artifacts into a clean "
            "persistent staging directory, validate Linux suites, and prove the "
            "checked-in artifacts/reference trees were not modified."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--hf-model-dir", type=Path, default=DEFAULT_HF_MODEL_DIR)
    parser.add_argument("--reference-python", type=Path, default=DEFAULT_REFERENCE_PYTHON)
    parser.add_argument("--pnnx", type=Path, default=DEFAULT_PNNX)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--clean-staging", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tree_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "fingerprint": None,
        }
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = file_path.stat()
        relative = file_path.relative_to(path).as_posix()
        total_bytes += stat.st_size
        file_count += 1
        digest.update(
            f"{relative}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8")
        )
    return {
        "path": str(path),
        "exists": True,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "fingerprint": digest.hexdigest(),
    }


def safe_clean(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    allowed = allowed_root.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise RuntimeError(f"Refusing to clean outside staging root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def run_capture(command: list[str], log_path: Path, cwd: Path = PROJECT_DIR) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    seconds = time.perf_counter() - start
    log_path.write_text(
        "$ " + " ".join(command) + "\n" + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed after {seconds:.1f}s with status "
            f"{completed.returncode}; see {log_path}"
        )
    return completed.stdout


def path_sha(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())


def dependency_environment(args: argparse.Namespace, logs_dir: Path) -> dict[str, object]:
    py_probe = (
        "import json, platform, sys; "
        "import numpy, PIL, torch, transformers; "
        "print(json.dumps({"
        "'python': sys.version.split()[0], "
        "'executable': sys.executable, "
        "'platform': platform.platform(), "
        "'numpy': numpy.__version__, "
        "'pillow': PIL.__version__, "
        "'torch': torch.__version__, "
        "'transformers': transformers.__version__"
        "}, sort_keys=True))"
    )
    py_output = run_capture(
        [str(args.reference_python), "-c", py_probe],
        logs_dir / "reference_python_environment.log",
    )
    cmake_output = run_capture(["cmake", "--version"], logs_dir / "cmake_version.log")
    return {
        "host_platform": platform.platform(),
        "python_reference": json.loads(py_output.splitlines()[-1]),
        "cmake_version": cmake_output.splitlines()[0] if cmake_output else None,
        "pnnx": path_sha(args.pnnx),
        "cli": path_sha(args.cli),
        "hf_model": {
            "path": str(args.hf_model_dir),
            "config_json": path_sha(args.hf_model_dir / "config.json"),
            "preprocessor_config_json": path_sha(
                args.hf_model_dir / "preprocessor_config.json"
            ),
            "tokenizer_json": path_sha(args.hf_model_dir / "tokenizer.json"),
            "tokenizer_config_json": path_sha(
                args.hf_model_dir / "tokenizer_config.json"
            ),
        },
    }


def git_status_for(paths: list[str], logs_dir: Path) -> str:
    return run_capture(
        ["git", "status", "--short", "--", *paths],
        logs_dir / "git_status_artifacts_reference.log",
    )


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    args.hf_model_dir = args.hf_model_dir.resolve()
    args.reference_python = args.reference_python.absolute()
    args.pnnx = args.pnnx.resolve()
    args.cli = args.cli.resolve()
    args.report = args.report.resolve()

    start = time.perf_counter()
    phase4j_report = args.work_dir / "phase4j_reference_capture_pipeline_report.json"
    logs_dir = args.work_dir / "logs" / "phase4k"
    if args.clean_staging:
        for child in (
            "reference-smoke-cpu-fp32",
            "reference-artifacts",
            "reference-docs",
            "dynamic-reference",
            "real-reference",
            "direct-staging-artifacts",
            "direct-export-docs",
            "logs",
        ):
            safe_clean(args.work_dir / child, args.work_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    before = {
        "artifacts": tree_snapshot(PROJECT_DIR / "artifacts"),
        "reference": tree_snapshot(PROJECT_DIR / "reference"),
        "git_status": git_status_for(["artifacts", "reference"], logs_dir),
    }
    report: dict[str, object] = {
        "phase": "4K",
        "status": "running",
        "started_at_unix": time.time(),
        "project_dir": str(PROJECT_DIR),
        "work_dir": str(args.work_dir),
        "existing_artifacts_reference_before": before,
        "failure_recovery_suggestions": [
            "Keep the existing artifacts/ and reference/ directories untouched; rerun with --clean-staging to remove only the Phase 4K staging subdirectories.",
            "If pnnx export fails, inspect logs/phase4j/direct-export/exporters under the Phase 4K work directory before retrying.",
            "If Linux runtime validation fails, inspect logs/phase4j/direct-export/validation and compare runtime_manifest.tsv between staging and artifacts.",
            "If dependency import checks fail, use the reference Python environment from /home/asus/work/hunyuanocr/.venv-reference and the pnnx executable from /home/asus/work/hunyuanocr/.venv-pnnx.",
        ],
    }
    try:
        report["dependency_environment"] = dependency_environment(args, logs_dir)
        free_before = shutil.disk_usage(args.work_dir.parent)

        command = [
            str(args.reference_python),
            "tools/reference/rebuild_references_and_artifacts.py",
            "--hf-model-dir",
            str(args.hf_model_dir),
            "--reference-python",
            str(args.reference_python),
            "--pnnx",
            str(args.pnnx),
            "--cli",
            str(args.cli),
            "--reference-dir",
            str(args.work_dir / "reference-smoke-cpu-fp32"),
            "--reference-artifacts-dir",
            str(args.work_dir / "reference-artifacts"),
            "--docs-dir",
            str(args.work_dir / "reference-docs"),
            "--direct-artifacts-dir",
            str(args.work_dir / "direct-staging-artifacts"),
            "--direct-export-docs-dir",
            str(args.work_dir / "direct-export-docs"),
            "--log-dir",
            str(args.work_dir / "logs" / "phase4j"),
            "--report",
            str(phase4j_report),
        ]
        if args.clean_staging:
            command.append("--clean-staging")
        if args.skip_validation:
            command.append("--skip-validation")
        run_capture(command, logs_dir / "phase4j_rebuild_driver.log")

        phase4j = json.loads(phase4j_report.read_text(encoding="utf-8"))
        if phase4j["status"] != "passed":
            raise RuntimeError(f"Phase 4J pipeline did not pass: {phase4j_report}")
        direct_report_path = Path(phase4j["direct_export"]["report"])
        direct_report = json.loads(direct_report_path.read_text(encoding="utf-8"))

        after = {
            "artifacts": tree_snapshot(PROJECT_DIR / "artifacts"),
            "reference": tree_snapshot(PROJECT_DIR / "reference"),
            "git_status": git_status_for(["artifacts", "reference"], logs_dir),
        }
        unchanged = before["artifacts"] == after["artifacts"] and before["reference"] == after["reference"]
        if not unchanged:
            raise RuntimeError("Existing artifacts/ or reference/ changed during acceptance")

        free_after = shutil.disk_usage(args.work_dir.parent)
        staging_artifacts = args.work_dir / "direct-staging-artifacts"
        report.update(
            {
                "status": "passed",
                "elapsed_seconds": time.perf_counter() - start,
                "clean_staging": bool(args.clean_staging),
                "phase4j_report": str(phase4j_report),
                "direct_export_report": str(direct_report_path),
                "linux_validation_count": len(direct_report.get("validations", [])),
                "manifest": direct_report["manifest"],
                "key_sha256": {
                    "staging_runtime_manifest": path_sha(
                        staging_artifacts / "runtime_manifest.tsv"
                    ),
                    "staging_runtime_compatibility": path_sha(
                        staging_artifacts / "runtime_compatibility.tsv"
                    ),
                    "existing_runtime_manifest": path_sha(
                        PROJECT_DIR / "artifacts/runtime_manifest.tsv"
                    ),
                    "phase4j_report": path_sha(phase4j_report),
                    "direct_export_report": path_sha(direct_report_path),
                },
                "disk_usage": {
                    "work_dir_bytes": du_bytes(args.work_dir),
                    "reference_dir_bytes": du_bytes(
                        args.work_dir / "reference-smoke-cpu-fp32"
                    ),
                    "direct_staging_artifacts_bytes": du_bytes(staging_artifacts),
                    "filesystem_free_bytes_before": free_before.free,
                    "filesystem_free_bytes_after": free_after.free,
                },
                "existing_artifacts_reference_after": after,
                "existing_artifacts_reference_unchanged": True,
            }
        )
        write_report(args.report, report)
        print(f"Phase 4K reproducible release acceptance passed: {args.report}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["elapsed_seconds"] = time.perf_counter() - start
        write_report(args.report, report)
        print(f"Phase 4K reproducible release acceptance failed: {args.report}")
        raise


if __name__ == "__main__":
    main()
