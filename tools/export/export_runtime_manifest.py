from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = PROJECT_DIR / "artifacts"
MANIFEST_PATH = MODEL_DIR / "runtime_manifest.tsv"
REPORT_PATH = PROJECT_DIR / "docs/runtime_manifest.json"


def component_files(name: str) -> list[Path]:
    directory = MODEL_DIR / name
    return [
        directory / f"{name}.ncnn.param",
        directory / f"{name}.ncnn.bin",
    ]


def required_files() -> list[Path]:
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
        files.extend(component_files(name))
    files.extend(
        [
            MODEL_DIR
            / "vision_patch_embed/vision_position_embedding.f32.bin",
            MODEL_DIR
            / "vision_patch_merger/vision_patch_merger_constants.f32.bin",
        ]
    )
    for layer in range(27):
        files.extend(component_files(f"vision_block{layer}"))
    for layer in range(24):
        files.extend(component_files(f"decoder_layer{layer}_prefill_kv"))
        files.extend(component_files(f"decoder_layer{layer}_decode_dynamic"))
    files.extend(
        [
            MODEL_DIR / "tokenizer/bytelevel_vocab.txt",
            MODEL_DIR / "tokenizer/bytelevel_bpe_merges.txt",
        ]
    )
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = required_files()
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing runtime model files: {missing}")
    entries = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(MODEL_DIR).as_posix()
        entry = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        entries.append(entry)
        print(f"[{index:03d}/{len(files):03d}] {relative}")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="ascii", newline="\n") as output:
        output.write("HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1\n")
        output.write(f"file_count\t{len(entries)}\n")
        for entry in entries:
            output.write(
                f"{entry['path']}\t{entry['bytes']}\t{entry['sha256']}\n"
            )
    report = {
        "format": "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1",
        "model_directory": "artifacts",
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "total_gib": sum(entry["bytes"] for entry in entries)
        / 1024**3,
        "manifest_path": MANIFEST_PATH.relative_to(PROJECT_DIR).as_posix(),
        "entries": entries,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Files: {len(entries)}")
    print(f"Total GiB: {report['total_gib']:.3f}")


if __name__ == "__main__":
    main()
