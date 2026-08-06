from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
SOURCE = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-official/HunyuanOCR_v1.0/assets/ie_parallel.jpg"
)
JPEG_TARGET = PROJECT_DIR / "tests/assets/ocr_receipt_real.jpg"
PNG_TARGET = PROJECT_DIR / "tests/assets/ocr_receipt_real.png"
REPORT = PROJECT_DIR / "tests/assets/real_ocr_assets.json"
SOURCE_SHA256 = "a69ce288471933fcf521ee950324338feebc07f7b329cf2896aa460bfada427d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("Official receipt source SHA-256 changed")
    JPEG_TARGET.write_bytes(SOURCE.read_bytes())
    with Image.open(SOURCE) as image:
        rgb = image.convert("RGB")
        rgb.save(PNG_TARGET, format="PNG", optimize=False)
        dimensions = list(rgb.size)
    report = {
        "format": "HUNYUANOCR_REAL_OCR_ASSETS_V1",
        "source": {
            "repository": "https://github.com/Tencent-Hunyuan/HunyuanOCR.git",
            "revision": "c55965d3da1e6f41987abec8068f2e70851318bc",
            "path": "HunyuanOCR_v1.0/assets/ie_parallel.jpg",
            "sha256": SOURCE_SHA256,
        },
        "dimensions": dimensions,
        "assets": [
            {
                "path": JPEG_TARGET.relative_to(PROJECT_DIR).as_posix(),
                "encoding": "original JPEG",
                "sha256": sha256(JPEG_TARGET),
            },
            {
                "path": PNG_TARGET.relative_to(PROJECT_DIR).as_posix(),
                "encoding": "Pillow-decoded RGB saved losslessly as PNG",
                "sha256": sha256(PNG_TARGET),
            },
        ],
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
