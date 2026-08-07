from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
RECOVERY_ROOT = Path.home() / "hunyuanocr-recovery" / "phase4i"
DEFAULT_HF_MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
DEFAULT_REFERENCE_DIR = PROJECT_DIR / "reference/smoke_en_cpu_fp32"
DEFAULT_STAGING_DIR = RECOVERY_ROOT / "direct-staging-artifacts"
DEFAULT_DOCS_DIR = RECOVERY_ROOT / "export-docs"
DEFAULT_LOG_DIR = RECOVERY_ROOT / "logs"
DEFAULT_REPORT = PROJECT_DIR / "docs/phase4i_direct_export_pipeline_report.json"
DEFAULT_REFERENCE_PYTHON = (
    Path.home() / "work/hunyuanocr/.venv-reference/bin/python"
)
DEFAULT_PNNX = Path.home() / "work/hunyuanocr/.venv-pnnx/bin/pnnx"
DEFAULT_CLI = (
    Path.home() / "hunyuanocr-recovery/phase4g/linux-release-build/hunyuanocr_cli"
)

VALIDATION_SUITES = ("smoke", "dynamic", "real-png", "real-jpeg", "cache-budgets")
HF_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def import_manifest_tools():
    sys.path.insert(0, str(PROJECT_DIR / "tools/export"))
    import export_runtime_manifest  # type: ignore[import-not-found]

    return export_runtime_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Directly rebuild HunyuanOCR-ncnn runtime artifacts into a staging "
            "directory by rerunning the pnnx/exporter scripts."
        )
    )
    parser.add_argument("--hf-model-dir", type=Path, default=DEFAULT_HF_MODEL_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reference-python", type=Path, default=DEFAULT_REFERENCE_PYTHON)
    parser.add_argument("--pnnx", type=Path, default=DEFAULT_PNNX)
    parser.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--clean-staging", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--runtime-verify",
        choices=("none", "size", "sha256"),
        default="size",
    )
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
    if allowed != resolved and allowed not in resolved.parents:
        raise RuntimeError(f"Refusing to clean outside staging root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    logger: PhaseLogger,
    label: str,
) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.write(f"Running {label}")
    logger.write(" ".join(command))
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
    log_path.write_text(completed.stdout, encoding="utf-8")
    seconds = time.perf_counter() - start
    tail = "\n".join(completed.stdout.splitlines()[-20:])
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


def parse_manifest(path: Path) -> dict[str, dict[str, object]]:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1":
        raise RuntimeError(f"Invalid manifest: {path}")
    entries: dict[str, dict[str, object]] = {}
    for line in lines[2:]:
        relative, size_text, digest = line.split("\t")
        entries[relative] = {
            "path": relative,
            "bytes": int(size_text),
            "sha256": digest,
        }
    return entries


def compare_manifest_structure(source_manifest: Path, staging_manifest: Path) -> dict[str, object]:
    source = parse_manifest(source_manifest)
    staging = parse_manifest(staging_manifest)
    source_paths = set(source)
    staging_paths = set(staging)
    changed = [
        path for path in sorted(source_paths & staging_paths)
        if source[path] != staging[path]
    ]
    return {
        "status": (
            "passed"
            if source_paths == staging_paths and len(staging) == len(source)
            else "failed"
        ),
        "source_file_count": len(source),
        "staging_file_count": len(staging),
        "missing": sorted(source_paths - staging_paths),
        "extra": sorted(staging_paths - source_paths),
        "changed_digest_or_size_count": len(changed),
        "changed_digest_or_size_first_20": changed[:20],
    }


def audit_inputs(args: argparse.Namespace, logger: PhaseLogger) -> dict[str, object]:
    require_dir(args.hf_model_dir, "local HuggingFace model directory")
    require_dir(args.reference_dir, "PyTorch reference tensor directory")
    require_file(args.reference_python, "reference Python")
    require_file(args.pnnx, "pnnx executable")
    require_file(args.cli, "Linux OCR CLI")
    for relative in HF_REQUIRED_FILES:
        require_file(args.hf_model_dir / relative, f"HF model file {relative}")
    require_file(PROJECT_DIR / "artifacts/runtime_manifest.tsv", "verified source manifest")

    completed = subprocess.run(
        [
            str(args.reference_python),
            "-c",
            (
                "import torch, transformers, numpy; "
                "print(torch.__version__); "
                "print(transformers.__version__); "
                "print(numpy.__version__)"
            ),
        ],
        cwd=PROJECT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Reference Python import check failed:\n{completed.stdout}")
    versions = completed.stdout.splitlines()
    logger.write("Input audit passed")
    logger.write(f"HF model dir      : {args.hf_model_dir}")
    logger.write(f"Reference tensors : {args.reference_dir}")
    logger.write(f"Staging directory : {args.staging_dir}")
    logger.write(f"Reference Python  : {args.reference_python}")
    logger.write(f"pnnx              : {args.pnnx}")
    logger.write(f"Runtime CLI       : {args.cli}")
    return {
        "hf_model_dir": str(args.hf_model_dir),
        "reference_dir": str(args.reference_dir),
        "staging_dir": str(args.staging_dir),
        "docs_dir": str(args.docs_dir),
        "reference_python": str(args.reference_python),
        "pnnx": str(args.pnnx),
        "cli": str(args.cli),
        "torch_version": versions[0] if len(versions) > 0 else None,
        "transformers_version": versions[1] if len(versions) > 1 else None,
        "numpy_version": versions[2] if len(versions) > 2 else None,
        "source_manifest_sha256_before": sha256_file(
            PROJECT_DIR / "artifacts/runtime_manifest.tsv"
        ),
    }


def exporter_base_args(args: argparse.Namespace) -> list[str]:
    return [
        "--model-dir",
        str(args.hf_model_dir),
        "--artifacts-dir",
        str(args.staging_dir),
        "--docs-dir",
        str(args.docs_dir),
    ]


def run_exporters(args: argparse.Namespace, logger: PhaseLogger) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    logs = args.log_dir / "exporters"
    py = str(args.reference_python)
    pnnx = str(args.pnnx)

    simple_exporters = [
        ("text_embedding", "tools/export/export_text_embedding.py", True),
        ("lm_head", "tools/export/export_lm_head.py", True),
        ("final_norm", "tools/export/export_final_norm.py", True),
        ("tokenizer", "tools/export/export_tokenizer_vocab.py", False),
    ]
    for label, script, needs_reference in simple_exporters:
        command = [py, script, *exporter_base_args(args)]
        if needs_reference:
            command.extend(["--reference-dir", str(args.reference_dir), "--pnnx", pnnx])
        results.append(
            run_command(command, PROJECT_DIR, logs / f"{label}.log", logger, label)
        )

    vision_command = [
        py,
        "tools/export/export_vision_tower_full.py",
        *exporter_base_args(args),
        "--reference-dir",
        str(args.reference_dir),
        "--pnnx",
        pnnx,
    ]
    results.append(
        run_command(
            vision_command,
            PROJECT_DIR,
            logs / "vision_tower_full.log",
            logger,
            "vision_tower_full",
        )
    )

    prefill_command = [
        py,
        "tools/export/export_decoder_prefill_kv.py",
        *exporter_base_args(args),
        "--reference-dir",
        str(args.reference_dir),
        "--pnnx",
        pnnx,
    ]
    results.append(
        run_command(
            prefill_command,
            PROJECT_DIR,
            logs / "decoder_prefill_kv_all.log",
            logger,
            "decoder_prefill_kv_all",
        )
    )

    for layer in range(24):
        dynamic_command = [
            py,
            "tools/export/export_decoder_dynamic.py",
            "--layer-index",
            str(layer),
            *exporter_base_args(args),
            "--reference-dir",
            str(args.reference_dir),
            "--pnnx",
            pnnx,
        ]
        results.append(
            run_command(
                dynamic_command,
                PROJECT_DIR,
                logs / f"decoder_dynamic_layer{layer:02d}.log",
                logger,
                f"decoder_dynamic_layer{layer:02d}",
            )
        )
    return results


def generate_manifest(args: argparse.Namespace, logger: PhaseLogger) -> dict[str, object]:
    manifest_tools = import_manifest_tools()
    manifest_report = args.log_dir / "direct_staging_runtime_manifest.json"
    report = manifest_tools.build_manifest(
        model_dir=args.staging_dir,
        manifest_path=args.staging_dir / "runtime_manifest.tsv",
        compatibility_path=args.staging_dir / "runtime_compatibility.tsv",
        report_path=manifest_report,
    )
    comparison = compare_manifest_structure(
        PROJECT_DIR / "artifacts/runtime_manifest.tsv",
        args.staging_dir / "runtime_manifest.tsv",
    )
    logger.write(f"Manifest structure comparison: {comparison['status']}")
    if comparison["status"] != "passed":
        raise RuntimeError("Direct staging manifest structure does not match runtime contract")
    return {
        "manifest": str(args.staging_dir / "runtime_manifest.tsv"),
        "compatibility": str(args.staging_dir / "runtime_compatibility.tsv"),
        "report": str(manifest_report),
        "file_count": report["file_count"],
        "total_gib": report["total_gib"],
        "comparison": comparison,
    }


def run_validation_suite(
    args: argparse.Namespace,
    suite: str,
    logger: PhaseLogger,
) -> dict[str, object]:
    suite_log_dir = args.log_dir / "validation" / suite
    command = [
        str(args.reference_python),
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
    return run_command(
        command,
        PROJECT_DIR,
        suite_log_dir / f"{suite}_driver.log",
        logger,
        f"validation_{suite}",
    )


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.hf_model_dir = args.hf_model_dir.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.staging_dir = args.staging_dir.resolve()
    args.docs_dir = args.docs_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    args.report = args.report.resolve()
    args.reference_python = args.reference_python.absolute()
    args.pnnx = args.pnnx.resolve()
    args.cli = args.cli.resolve()
    staging_root = args.staging_dir.parent

    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger = PhaseLogger(args.log_dir / "phase4i_direct_export_pipeline.log")
    start = time.perf_counter()
    report: dict[str, object] = {
        "phase": "4I",
        "status": "running",
        "started_at_unix": time.time(),
    }
    try:
        logger.write("HunyuanOCR-ncnn Phase 4I direct export pipeline")
        report["audit"] = audit_inputs(args, logger)
        if args.clean_staging:
            safe_clean(args.staging_dir, staging_root)
            safe_clean(args.docs_dir, staging_root)
        args.staging_dir.mkdir(parents=True, exist_ok=True)
        args.docs_dir.mkdir(parents=True, exist_ok=True)

        report["exports"] = run_exporters(args, logger)
        report["manifest"] = generate_manifest(args, logger)

        validations = []
        if not args.skip_validation:
            for suite in VALIDATION_SUITES:
                validations.append(run_validation_suite(args, suite, logger))
        report["validations"] = validations
        report["source_manifest_sha256_after"] = sha256_file(
            PROJECT_DIR / "artifacts/runtime_manifest.tsv"
        )
        report["source_manifest_unchanged"] = (
            report["source_manifest_sha256_after"]
            == report["audit"]["source_manifest_sha256_before"]
        )
        if not report["source_manifest_unchanged"]:
            raise RuntimeError("Existing artifacts manifest changed unexpectedly")
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
