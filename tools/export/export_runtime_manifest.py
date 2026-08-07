from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = PROJECT_DIR / "artifacts"
MANIFEST_PATH = MODEL_DIR / "runtime_manifest.tsv"
COMPATIBILITY_PATH = MODEL_DIR / "runtime_compatibility.tsv"
REPORT_PATH = PROJECT_DIR / "docs/runtime_manifest.json"
RUNTIME_VERSION = "0.1.0"


def component_files(model_dir: Path, name: str) -> list[Path]:
    directory = model_dir / name
    return [
        directory / f"{name}.ncnn.param",
        directory / f"{name}.ncnn.bin",
    ]


def required_files(model_dir: Path = MODEL_DIR) -> list[Path]:
    files: list[Path] = []
    for name in [
        "final_norm",
        "lm_head",
        "text_embedding",
        "vision_patch_embed",
        "vision_patch_merger_pre_rms",
        "vision_patch_merger_conv",
        "vision_patch_merger_projection",
        "vision_patch_merger_post_rms",
    ]:
        files.extend(component_files(model_dir, name))
    files.extend(
        [
            model_dir
            / "vision_patch_embed/vision_position_embedding.f32.bin",
            model_dir
            / "vision_patch_merger/vision_patch_merger_constants.f32.bin",
        ]
    )
    for layer in range(27):
        files.extend(component_files(model_dir, f"vision_block{layer}"))
    for layer in range(24):
        files.extend(component_files(model_dir, f"decoder_layer{layer}_prefill_kv"))
        files.extend(component_files(model_dir, f"decoder_layer{layer}_decode_dynamic"))
    files.extend(
        [
            model_dir / "tokenizer/bytelevel_vocab.txt",
            model_dir / "tokenizer/bytelevel_bpe_merges.txt",
        ]
    )
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    model_dir: Path = MODEL_DIR,
    manifest_path: Path | None = None,
    compatibility_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    model_dir = model_dir.resolve()
    manifest_path = manifest_path or model_dir / "runtime_manifest.tsv"
    compatibility_path = compatibility_path or model_dir / "runtime_compatibility.tsv"
    report_path = report_path or REPORT_PATH
    files = required_files(model_dir)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing runtime model files: {missing}")
    entries = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(model_dir).as_posix()
        entry = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        entries.append(entry)
        print(f"[{index:03d}/{len(files):03d}] {relative}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="ascii", newline="\n") as output:
        output.write("HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1\n")
        output.write(f"file_count\t{len(entries)}\n")
        for entry in entries:
            output.write(
                f"{entry['path']}\t{entry['bytes']}\t{entry['sha256']}\n"
            )
    compatibility = {
        "format": "HUNYUANOCR_NCNN_RUNTIME_COMPATIBILITY_V1",
        "model_id": "tencent/HunyuanOCR",
        "runtime_abi_major": "0",
        "runtime_min_version": RUNTIME_VERSION,
        "runtime_max_exclusive_version": "1.0.0",
        "manifest_format": "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1",
        "file_count": str(len(entries)),
        "precision": "fp32",
        "jpeg_pixel_contract": "stb_rgb_v1",
    }
    with compatibility_path.open("w", encoding="ascii", newline="\n") as output:
        output.write(f"{compatibility['format']}\n")
        for key in (
            "model_id",
            "runtime_abi_major",
            "runtime_min_version",
            "runtime_max_exclusive_version",
            "manifest_format",
            "file_count",
            "precision",
            "jpeg_pixel_contract",
        ):
            output.write(f"{key}\t{compatibility[key]}\n")
    report = {
        "format": "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1",
        "runtime_compatibility": compatibility,
        "model_directory": str(model_dir),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "total_gib": sum(entry["bytes"] for entry in entries)
        / 1024**3,
        "manifest_path": str(manifest_path),
        "entries": entries,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    print(f"Compatibility: {compatibility_path}")
    print(f"Files: {len(entries)}")
    print(f"Total GiB: {report['total_gib']:.3f}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the HunyuanOCR-ncnn runtime manifest."
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--compatibility", type=Path)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(
        model_dir=args.model_dir,
        manifest_path=args.manifest,
        compatibility_path=args.compatibility,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
