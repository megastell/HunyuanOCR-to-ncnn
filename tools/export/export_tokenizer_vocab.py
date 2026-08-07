from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUT_PATH = ARTIFACTS_DIR / "tokenizer/bytelevel_vocab.txt"
MERGES_PATH = ARTIFACTS_DIR / "tokenizer/bytelevel_bpe_merges.txt"
REPORT_PATH = DOCS_DIR / "tokenizer_vocab_export.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tokenizer byte-level data.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global MODEL_DIR, ARTIFACTS_DIR, DOCS_DIR, OUTPUT_PATH, MERGES_PATH, REPORT_PATH
    MODEL_DIR = args.model_dir.resolve()
    ARTIFACTS_DIR = args.artifacts_dir.resolve()
    DOCS_DIR = args.docs_dir.resolve()
    OUTPUT_PATH = ARTIFACTS_DIR / "tokenizer/bytelevel_vocab.txt"
    MERGES_PATH = ARTIFACTS_DIR / "tokenizer/bytelevel_bpe_merges.txt"
    REPORT_PATH = DOCS_DIR / "tokenizer_vocab_export.json"

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    vocabulary = tokenizer.get_vocab()
    maximum_id = max(int(token_id) for token_id in vocabulary.values())
    tokens: list[str | None] = [None] * (maximum_id + 1)
    for token, token_id in vocabulary.items():
        index = int(token_id)
        if tokens[index] is not None and tokens[index] != token:
            raise RuntimeError(f"Duplicate tokenizer id: {index}")
        tokens[index] = token

    missing_ids = [index for index, token in enumerate(tokens) if token is None]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="ascii", newline="\n") as output:
        output.write("HUNYUANOCR_BYTELEVEL_VOCAB_V1\n")
        output.write(f"{len(tokens)}\n")
        for token in tokens:
            encoded = "" if token is None else token.encode("utf-8").hex()
            output.write(encoded + "\n")

    tokenizer_data = json.loads(
        (MODEL_DIR / "tokenizer.json").read_text(encoding="utf-8")
    )
    merges = tokenizer_data["model"]["merges"]
    with MERGES_PATH.open("w", encoding="ascii", newline="\n") as output:
        output.write("HUNYUANOCR_BYTELEVEL_BPE_MERGES_V1\n")
        output.write(f"{len(merges)}\n")
        for left, right in merges:
            output.write(
                f"{left.encode('utf-8').hex()}\t"
                f"{right.encode('utf-8').hex()}\n"
            )

    smoke_ids = [
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
    ]
    smoke_text = tokenizer.decode(
        smoke_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    report = {
        "format": "HUNYUANOCR_BYTELEVEL_VOCAB_V1",
        "vocabulary_size": len(tokens),
        "missing_id_count": len(missing_ids),
        "missing_ids": missing_ids,
        "eos_token_id": int(tokenizer.eos_token_id),
        "smoke_token_ids": smoke_ids,
        "smoke_text": smoke_text,
        "output_path": str(OUTPUT_PATH),
        "output_bytes": OUTPUT_PATH.stat().st_size,
        "merges_format": "HUNYUANOCR_BYTELEVEL_BPE_MERGES_V1",
        "merge_count": len(merges),
        "merges_path": str(MERGES_PATH),
        "merges_bytes": MERGES_PATH.stat().st_size,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Vocabulary entries: {len(tokens)}")
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Smoke text: {smoke_text!r}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Merges: {MERGES_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
