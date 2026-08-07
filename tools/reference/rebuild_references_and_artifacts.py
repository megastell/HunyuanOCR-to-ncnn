from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
RECOVERY_ROOT = Path.home() / "hunyuanocr-recovery" / "phase4j"
DEFAULT_HF_MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
DEFAULT_REFERENCE_PYTHON = (
    Path.home() / "work/hunyuanocr/.venv-reference/bin/python"
)
DEFAULT_PNNX = Path.home() / "work/hunyuanocr/.venv-pnnx/bin/pnnx"
DEFAULT_CLI = (
    Path.home() / "hunyuanocr-recovery/phase4g/linux-release-build/hunyuanocr_cli"
)
DEFAULT_REFERENCE_DIR = RECOVERY_ROOT / "reference-smoke-cpu-fp32"
DEFAULT_REFERENCE_ARTIFACTS_DIR = RECOVERY_ROOT / "reference-artifacts"
DEFAULT_DOCS_DIR = RECOVERY_ROOT / "reference-docs"
DEFAULT_DIRECT_ARTIFACTS_DIR = RECOVERY_ROOT / "direct-staging-artifacts"
DEFAULT_DIRECT_EXPORT_DOCS_DIR = RECOVERY_ROOT / "direct-export-docs"
DEFAULT_LOG_DIR = RECOVERY_ROOT / "logs"
DEFAULT_REPORT = PROJECT_DIR / "docs/phase4j_reference_capture_pipeline_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate PyTorch reference tensors from local images/model, then "
            "run the Phase 4I direct pnnx artifact export and Linux validation."
        )
    )
    parser.add_argument("--hf-model-dir", type=Path, default=DEFAULT_HF_MODEL_DIR)
    parser.add_argument("--reference-python", type=Path, default=DEFAULT_REFERENCE_PYTHON)
    parser.add_argument("--pnnx", type=Path, default=DEFAULT_PNNX)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument(
        "--reference-artifacts-dir",
        type=Path,
        default=DEFAULT_REFERENCE_ARTIFACTS_DIR,
    )
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument(
        "--direct-artifacts-dir",
        type=Path,
        default=DEFAULT_DIRECT_ARTIFACTS_DIR,
    )
    parser.add_argument(
        "--direct-export-docs-dir",
        type=Path,
        default=DEFAULT_DIRECT_EXPORT_DOCS_DIR,
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--clean-staging", action="store_true")
    parser.add_argument("--skip-direct-export", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


class PhaseLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("w", encoding="utf-8", newline="\n")

    def close(self) -> None:
        self.stream.close()

    def write(self, message: str = "") -> None:
        print(message)
        self.stream.write(message + "\n")
        self.stream.flush()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"Missing {label}: {path}")


def safe_clean(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    allowed = allowed_root.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise RuntimeError(f"Refusing to clean outside Phase 4J root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    logger: PhaseLogger,
    label: str,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.write(f"Running {label}")
    logger.write(" ".join(command))
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    seconds = time.perf_counter() - start
    tail = "\n".join(completed.stdout.splitlines()[-24:])
    if tail:
        logger.write(tail)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with status {completed.returncode}: {log_path}")
    logger.write(f"Completed {label} in {seconds:.1f}s")
    return {
        "label": label,
        "status": "passed",
        "seconds": seconds,
        "log": str(log_path),
    }


def audit_inputs(args: argparse.Namespace, logger: PhaseLogger) -> dict[str, object]:
    require_dir(args.hf_model_dir, "local HuggingFace model directory")
    require_file(args.reference_python, "reference Python")
    require_file(args.pnnx, "pnnx executable")
    require_file(args.cli, "Linux OCR CLI")
    for relative in (
        "tests/assets/ocr_smoke_en.png",
        "tests/assets/ocr_wide_en.png",
        "tests/assets/ocr_square_en.png",
        "tests/assets/ocr_tall_en.png",
        "tests/assets/ocr_receipt_real.png",
        "tests/assets/ocr_receipt_real.jpg",
        "tests/assets/ocr_receipt_real_stb.ppm",
    ):
        require_file(PROJECT_DIR / relative, relative)
    logger.write("Input audit passed")
    logger.write(f"HF model dir      : {args.hf_model_dir}")
    logger.write(f"Reference dir     : {args.reference_dir}")
    logger.write(f"Direct artifacts  : {args.direct_artifacts_dir}")
    return {
        "hf_model_dir": str(args.hf_model_dir),
        "reference_python": str(args.reference_python),
        "pnnx": str(args.pnnx),
        "cli": str(args.cli),
        "reference_dir": str(args.reference_dir),
        "reference_artifacts_dir": str(args.reference_artifacts_dir),
        "docs_dir": str(args.docs_dir),
        "direct_artifacts_dir": str(args.direct_artifacts_dir),
        "direct_export_docs_dir": str(args.direct_export_docs_dir),
    }


def reference_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HUNYUANOCR_MODEL_DIR": str(args.hf_model_dir),
            "HUNYUANOCR_REFERENCE_DIR": str(args.reference_dir),
            "HUNYUANOCR_ARTIFACTS_DIR": str(args.reference_artifacts_dir),
            "HUNYUANOCR_DOCS_DIR": str(args.docs_dir),
            "HUNYUANOCR_SMOKE_IMAGE": str(PROJECT_DIR / "tests/assets/ocr_smoke_en.png"),
            "HUNYUANOCR_DYNAMIC_REFERENCE_DIR": str(
                RECOVERY_ROOT / "dynamic-reference"
            ),
            "HUNYUANOCR_REAL_REFERENCE_DIR": str(RECOVERY_ROOT / "real-reference"),
            "HUNYUANOCR_RECOVERY_DIR": str(RECOVERY_ROOT),
        }
    )
    return env


def capture_references(
    args: argparse.Namespace,
    logger: PhaseLogger,
) -> list[dict[str, object]]:
    py = str(args.reference_python)
    logs = args.log_dir / "reference-capture"
    env = reference_env(args)
    steps = [
        ("smoke_reference", [py, "tools/reference/run_reference_smoke.py"]),
        ("split_contract", [py, "tools/reference/capture_split_contract.py"]),
        ("dynamic_expected", [py, "tools/reference/capture_dynamic_image_regressions.py"]),
        ("real_expected", [py, "tools/reference/capture_real_ocr_reference.py"]),
        ("vision_tower_full", [py, "tools/reference/capture_vision_tower_full.py"]),
        (
            "decoder_decode_steps_1_3",
            [
                py,
                "tools/reference/capture_decoder_autoregressive_steps.py",
                "--start-step",
                "1",
                "--end-step",
                "3",
            ],
        ),
        (
            "decoder_prefill_kv",
            [py, "tools/reference/capture_decoder_24_layer_prefill.py"],
        ),
    ]
    results = []
    for label, command in steps:
        results.append(
            run_command(
                command,
                PROJECT_DIR,
                logs / f"{label}.log",
                logger,
                label,
                env=env,
            )
        )
    return results


def summarize_reference_dir(reference_dir: Path) -> dict[str, object]:
    files = sorted(path for path in reference_dir.rglob("*.npy") if path.is_file())
    return {
        "path": str(reference_dir),
        "npy_file_count": len(files),
        "first_20": [path.relative_to(reference_dir).as_posix() for path in files[:20]],
    }


def run_direct_export(
    args: argparse.Namespace,
    logger: PhaseLogger,
) -> dict[str, object]:
    report_path = PROJECT_DIR / "docs/phase4j_direct_export_pipeline_report.json"
    command = [
        str(args.reference_python),
        "tools/export/rebuild_runtime_artifacts.py",
        "--hf-model-dir",
        str(args.hf_model_dir),
        "--reference-dir",
        str(args.reference_dir),
        "--staging-dir",
        str(args.direct_artifacts_dir),
        "--docs-dir",
        str(args.direct_export_docs_dir),
        "--log-dir",
        str(args.log_dir / "direct-export"),
        "--report",
        str(report_path),
        "--reference-python",
        str(args.reference_python),
        "--pnnx",
        str(args.pnnx),
        "--cli",
        str(args.cli),
    ]
    if args.skip_validation:
        command.append("--skip-validation")
    result = run_command(
        command,
        PROJECT_DIR,
        args.log_dir / "direct-export" / "phase4i_driver.log",
        logger,
        "phase4i_direct_export_with_phase4j_references",
    )
    direct_report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        **result,
        "report": str(report_path),
        "direct_status": direct_report["status"],
        "file_count": direct_report["manifest"]["file_count"],
        "total_gib": direct_report["manifest"]["total_gib"],
        "manifest_comparison": direct_report["manifest"]["comparison"],
        "validation_count": len(direct_report.get("validations", [])),
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
    args.reference_python = args.reference_python.absolute()
    args.pnnx = args.pnnx.resolve()
    args.cli = args.cli.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.reference_artifacts_dir = args.reference_artifacts_dir.resolve()
    args.docs_dir = args.docs_dir.resolve()
    args.direct_artifacts_dir = args.direct_artifacts_dir.resolve()
    args.direct_export_docs_dir = args.direct_export_docs_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    args.report = args.report.resolve()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger = PhaseLogger(args.log_dir / "phase4j_reference_capture_pipeline.log")
    start = time.perf_counter()
    report: dict[str, object] = {
        "phase": "4J",
        "status": "running",
        "started_at_unix": time.time(),
    }
    try:
        logger.write("HunyuanOCR-ncnn Phase 4J reference capture pipeline")
        report["audit"] = audit_inputs(args, logger)
        if args.clean_staging:
            for path in (
                args.reference_dir,
                args.reference_artifacts_dir,
                args.docs_dir,
                args.direct_artifacts_dir,
                args.direct_export_docs_dir,
                args.log_dir / "direct-export",
                args.log_dir / "reference-capture",
                RECOVERY_ROOT / "dynamic-reference",
                RECOVERY_ROOT / "real-reference",
            ):
                safe_clean(path, RECOVERY_ROOT)
        args.reference_dir.mkdir(parents=True, exist_ok=True)
        args.reference_artifacts_dir.mkdir(parents=True, exist_ok=True)
        args.docs_dir.mkdir(parents=True, exist_ok=True)

        report["reference_capture"] = capture_references(args, logger)
        report["reference_summary"] = summarize_reference_dir(args.reference_dir)
        if not args.skip_direct_export:
            report["direct_export"] = run_direct_export(args, logger)
        report["status"] = "passed"
        report["elapsed_seconds"] = time.perf_counter() - start
        write_report(args.report, report)
        logger.write(f"Report: {args.report}")
        logger.write("Pipeline status: passed")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["elapsed_seconds"] = time.perf_counter() - start
        write_report(args.report, report)
        logger.write("Pipeline status: failed")
        logger.write(str(exc))
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
