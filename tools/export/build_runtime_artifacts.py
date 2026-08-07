from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
RECOVERY_ROOT = Path.home() / "hunyuanocr-recovery" / "phase4h"
DEFAULT_HF_MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
DEFAULT_SOURCE_ARTIFACTS = PROJECT_DIR / "artifacts"
DEFAULT_STAGING_DIR = RECOVERY_ROOT / "staging-artifacts"
DEFAULT_LOG_DIR = RECOVERY_ROOT / "logs"
DEFAULT_REPORT = PROJECT_DIR / "docs/phase4h_runtime_artifact_pipeline_report.json"
DEFAULT_CLI = (
    Path.home() / "hunyuanocr-recovery/phase4g/linux-release-build/hunyuanocr_cli"
)
PNNX_PATH = Path.home() / "work/hunyuanocr/.venv-pnnx/bin/pnnx"
NCNN_CMAKE_DIR = Path.home() / ".local/ncnn-cpu-ropefix-rmsnorm/lib/cmake/ncnn"

VALIDATION_SUITES = ("smoke", "dynamic", "real-png", "real-jpeg", "cache-budgets")
HF_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXPORT_SCRIPTS = (
    "tools/export/export_decoder_dynamic.py",
    "tools/export/export_decoder_prefill_kv.py",
    "tools/export/export_final_norm.py",
    "tools/export/export_lm_head.py",
    "tools/export/export_runtime_manifest.py",
    "tools/export/export_text_embedding.py",
    "tools/export/export_tokenizer_vocab.py",
    "tools/export/export_vision_tower_full.py",
)
REFERENCE_SCRIPTS = (
    "tools/reference/capture_dynamic_image_regressions.py",
    "tools/reference/capture_real_ocr_reference.py",
    "tools/reference/generate_dynamic_ocr_cases.py",
    "tools/reference/prepare_real_ocr_assets.py",
    "tools/reference/run_reference_smoke.py",
)
NCNN_PATCH_MARKERS = (
    "ncnn_fp32_rope",
    "RMSNorm",
    "MatMul",
)


def import_manifest_tools():
    sys.path.insert(0, str(PROJECT_DIR / "tools/export"))
    import export_runtime_manifest  # type: ignore[import-not-found]

    return export_runtime_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible staging runtime artifact directory from local "
            "HunyuanOCR conversion outputs."
        )
    )
    parser.add_argument("--hf-model-dir", type=Path, default=DEFAULT_HF_MODEL_DIR)
    parser.add_argument("--source-artifacts", type=Path, default=DEFAULT_SOURCE_ARTIFACTS)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument(
        "--runtime-verify",
        choices=("none", "size", "sha256"),
        default="size",
        help="Manifest verification mode used by the OCR runtime validation CLI.",
    )
    parser.add_argument(
        "--clean-staging",
        action="store_true",
        help="Remove and recreate the staging directory before copying artifacts.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Generate and compare artifacts without running OCR validation suites.",
    )
    return parser.parse_args()


class PhaseLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.log_path.open("w", encoding="utf-8", newline="\n")

    def close(self) -> None:
        self._stream.close()

    def write(self, message: str = "") -> None:
        print(message)
        self._stream.write(message + "\n")
        self._stream.flush()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"Missing {label}: {path}")


def safe_clean_dir(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    allowed = allowed_root.resolve()
    if allowed != resolved and allowed not in resolved.parents:
        raise RuntimeError(
            f"Refusing to clean staging outside persistent Phase 4H root: {resolved}"
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def parse_manifest(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 2 or lines[0] != "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1":
        raise RuntimeError(f"Invalid runtime manifest header: {path}")
    key, count_text = lines[1].split("\t", 1)
    if key != "file_count":
        raise RuntimeError(f"Invalid file_count line in manifest: {path}")
    entries = []
    for line in lines[2:]:
        relative, size_text, digest = line.split("\t")
        entries.append(
            {
                "path": relative,
                "bytes": int(size_text),
                "sha256": digest,
            }
        )
    declared_count = int(count_text)
    if declared_count != len(entries):
        raise RuntimeError(
            f"Manifest count mismatch in {path}: {declared_count} != {len(entries)}"
        )
    return {
        "path": str(path),
        "file_count": len(entries),
        "entries": entries,
        "by_path": {entry["path"]: entry for entry in entries},
    }


def compare_manifests(source: Path, staging: Path) -> dict[str, object]:
    source_manifest = parse_manifest(source)
    staging_manifest = parse_manifest(staging)
    source_paths = set(source_manifest["by_path"])
    staging_paths = set(staging_manifest["by_path"])
    missing = sorted(source_paths - staging_paths)
    extra = sorted(staging_paths - source_paths)
    changed = []
    for relative in sorted(source_paths & staging_paths):
        source_entry = source_manifest["by_path"][relative]
        staging_entry = staging_manifest["by_path"][relative]
        if source_entry != staging_entry:
            changed.append(
                {
                    "path": relative,
                    "source": source_entry,
                    "staging": staging_entry,
                }
            )
    status = not missing and not extra and not changed
    return {
        "status": "passed" if status else "failed",
        "source_file_count": source_manifest["file_count"],
        "staging_file_count": staging_manifest["file_count"],
        "missing": missing,
        "extra": extra,
        "changed": changed,
    }


def audit_inputs(args: argparse.Namespace, logger: PhaseLogger) -> dict[str, object]:
    require_dir(args.hf_model_dir, "local HuggingFace model directory")
    require_dir(args.source_artifacts, "source artifacts directory")
    require_file(args.cli, "Linux OCR CLI")

    hf_files = {}
    for relative in HF_REQUIRED_FILES:
        path = args.hf_model_dir / relative
        require_file(path, f"HF model file {relative}")
        hf_files[relative] = path.stat().st_size

    script_status = {}
    for relative in EXPORT_SCRIPTS + REFERENCE_SCRIPTS:
        path = PROJECT_DIR / relative
        require_file(path, relative)
        script_status[relative] = path.stat().st_size

    toolchain = {
        "pnnx": {"path": str(PNNX_PATH), "exists": PNNX_PATH.is_file()},
        "ncnn_cmake_dir": {
            "path": str(NCNN_CMAKE_DIR),
            "exists": NCNN_CMAKE_DIR.is_dir(),
        },
    }
    marker_hits = {}
    ncnn_root = Path.home() / "work/hunyuanocr/ncnn"
    search_tool = shutil.which("rg")
    for marker in NCNN_PATCH_MARKERS:
        if ncnn_root.is_dir() and search_tool is not None:
            completed = subprocess.run(
                [
                    search_tool,
                    "-n",
                    marker,
                    str(ncnn_root / "src"),
                    str(ncnn_root / "tools"),
                ],
                cwd=PROJECT_DIR,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            marker_hits[marker] = len(
                [line for line in completed.stdout.splitlines() if line.strip()]
            )
        else:
            marker_hits[marker] = None

    logger.write("Input audit passed")
    logger.write(f"HF model dir      : {args.hf_model_dir}")
    logger.write(f"Source artifacts  : {args.source_artifacts}")
    logger.write(f"Staging directory : {args.staging_dir}")
    logger.write(f"Runtime CLI       : {args.cli}")
    logger.write(f"pnnx available    : {toolchain['pnnx']['exists']}")
    logger.write(f"ncnn CMake dir    : {toolchain['ncnn_cmake_dir']['exists']}")

    return {
        "hf_model_dir": str(args.hf_model_dir),
        "hf_required_files": hf_files,
        "source_artifacts": str(args.source_artifacts),
        "staging_dir": str(args.staging_dir),
        "cli": str(args.cli),
        "export_scripts": script_status,
        "toolchain": toolchain,
        "ncnn_precision_marker_hits": marker_hits,
    }


def copy_runtime_files(
    source_artifacts: Path,
    staging_dir: Path,
    logger: PhaseLogger,
) -> dict[str, object]:
    manifest_tools = import_manifest_tools()
    required_files = manifest_tools.required_files(source_artifacts.resolve())
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing source runtime artifacts: {missing}")

    copied = []
    total_bytes = 0
    for index, source in enumerate(required_files, start=1):
        relative = source.relative_to(source_artifacts).as_posix()
        destination = staging_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        size = destination.stat().st_size
        total_bytes += size
        copied.append({"path": relative, "bytes": size})
        logger.write(f"[copy {index:03d}/{len(required_files):03d}] {relative}")
    return {
        "file_count": len(copied),
        "total_bytes": total_bytes,
        "total_gib": total_bytes / 1024**3,
        "files": copied,
    }


def generate_staging_manifest(
    staging_dir: Path,
    log_dir: Path,
    logger: PhaseLogger,
) -> dict[str, object]:
    manifest_tools = import_manifest_tools()
    report_path = log_dir / "staging_runtime_manifest.json"
    report = manifest_tools.build_manifest(
        model_dir=staging_dir,
        manifest_path=staging_dir / "runtime_manifest.tsv",
        compatibility_path=staging_dir / "runtime_compatibility.tsv",
        report_path=report_path,
    )
    logger.write(f"Generated staging manifest: {staging_dir / 'runtime_manifest.tsv'}")
    logger.write(f"Generated staging compatibility: {staging_dir / 'runtime_compatibility.tsv'}")
    return {
        "manifest": str(staging_dir / "runtime_manifest.tsv"),
        "compatibility": str(staging_dir / "runtime_compatibility.tsv"),
        "report": str(report_path),
        "file_count": report["file_count"],
        "total_gib": report["total_gib"],
    }


def run_validation_suite(
    args: argparse.Namespace,
    suite: str,
    logger: PhaseLogger,
) -> dict[str, object]:
    suite_log_dir = args.log_dir / "validation" / suite
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_DIR / "tools/validate/validate_runtime_suite.py"),
        "--cli",
        str(args.cli),
        "--model-dir",
        str(args.staging_dir),
        "--log-dir",
        str(suite_log_dir),
        "--suite",
        suite,
        "--packing",
        "both",
        "--verify",
        args.runtime_verify,
    ]
    logger.write(f"Running validation suite: {suite}")
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
    output_log = suite_log_dir / f"{suite}_driver.log"
    output_log.write_text(completed.stdout, encoding="utf-8")
    logger.write(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(f"Validation suite failed: {suite}; see {output_log}")
    summary = suite_log_dir / f"{suite}_summary.json"
    require_file(summary, f"{suite} validation summary")
    return {
        "suite": suite,
        "status": "passed",
        "seconds": time.perf_counter() - start,
        "driver_log": str(output_log),
        "summary": str(summary),
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.hf_model_dir = args.hf_model_dir.resolve()
    args.source_artifacts = args.source_artifacts.resolve()
    args.staging_dir = args.staging_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    args.report = args.report.resolve()
    args.cli = args.cli.resolve()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger = PhaseLogger(args.log_dir / "phase4h_runtime_artifacts_pipeline.log")
    start = time.perf_counter()
    report: dict[str, object] = {
        "phase": "4H",
        "status": "running",
        "started_at_unix": time.time(),
    }
    try:
        logger.write("HunyuanOCR-ncnn Phase 4H runtime artifact pipeline")
        audit = audit_inputs(args, logger)
        report["audit"] = audit

        if args.clean_staging:
            safe_clean_dir(args.staging_dir, RECOVERY_ROOT)
        args.staging_dir.mkdir(parents=True, exist_ok=True)

        copy_report = copy_runtime_files(args.source_artifacts, args.staging_dir, logger)
        report["copy"] = copy_report

        staging_manifest = generate_staging_manifest(args.staging_dir, args.log_dir, logger)
        report["staging_manifest"] = staging_manifest

        comparison = compare_manifests(
            args.source_artifacts / "runtime_manifest.tsv",
            args.staging_dir / "runtime_manifest.tsv",
        )
        report["manifest_comparison"] = comparison
        logger.write(f"Manifest comparison: {comparison['status']}")
        if comparison["status"] != "passed":
            raise RuntimeError("Staging manifest differs from verified artifacts")

        validations = []
        if not args.skip_validation:
            for suite in VALIDATION_SUITES:
                validations.append(run_validation_suite(args, suite, logger))
        report["validations"] = validations

        report["status"] = "passed"
        report["elapsed_seconds"] = time.perf_counter() - start
        write_report(args.report, report)
        logger.write(f"Report: {args.report}")
        logger.write(f"Pipeline status: {report['status']}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["elapsed_seconds"] = time.perf_counter() - start
        write_report(args.report, report)
        logger.write(f"Pipeline status: failed")
        logger.write(str(exc))
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
