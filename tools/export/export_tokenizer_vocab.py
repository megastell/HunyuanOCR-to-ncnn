from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
MODEL_DIR = Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"
OUTPUT_PATH = PROJECT_DIR / "artifacts/tokenizer/bytelevel_vocab.txt"
REPORT_PATH = PROJECT_DIR / "docs/tokenizer_vocab_export.json"


def main() -> None:
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
        "output_path": OUTPUT_PATH.relative_to(PROJECT_DIR).as_posix(),
        "output_bytes": OUTPUT_PATH.stat().st_size,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Vocabulary entries: {len(tokens)}")
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Smoke text: {smoke_text!r}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
