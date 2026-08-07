from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageChops


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
SOURCE = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-official/HunyuanOCR_v1.0/assets/ie_parallel.jpg"
)
JPEG_TARGET = PROJECT_DIR / "tests/assets/ocr_receipt_real.jpg"
PNG_TARGET = PROJECT_DIR / "tests/assets/ocr_receipt_real.png"
STB_RGB_TARGET = PROJECT_DIR / "tests/assets/ocr_receipt_real_stb.ppm"
REPORT = PROJECT_DIR / "tests/assets/real_ocr_assets.json"
SOURCE_SHA256 = "a69ce288471933fcf521ee950324338feebc07f7b329cf2896aa460bfada427d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder",
        type=Path,
        default=PROJECT_DIR / "build-phase4e/hunyuanocr_decode_image_rgb",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("Official receipt source SHA-256 changed")
    if not args.decoder.is_file():
        raise RuntimeError(f"Production image decoder not found: {args.decoder}")
    JPEG_TARGET.write_bytes(SOURCE.read_bytes())
    with Image.open(SOURCE) as image:
        rgb = image.convert("RGB")
        rgb.save(PNG_TARGET, format="PNG", optimize=False)
        dimensions = list(rgb.size)
    completed = subprocess.run(
        [str(args.decoder), str(JPEG_TARGET), str(STB_RGB_TARGET)],
        check=True,
        text=True,
        capture_output=True,
    )
    with (
        Image.open(PNG_TARGET) as pillow_image,
        Image.open(STB_RGB_TARGET) as stb_image,
    ):
        pillow_rgb = pillow_image.convert("RGB")
        stb_rgb = stb_image.convert("RGB")
        if pillow_rgb.size != stb_rgb.size:
            raise RuntimeError("Pillow and stb_image dimensions differ")
        difference = ImageChops.difference(pillow_rgb, stb_rgb)
        extrema = difference.getextrema()
        red, green, blue = difference.split()
        difference_mask = ImageChops.lighter(ImageChops.lighter(red, green), blue)
        differing_pixels = dimensions[0] * dimensions[1]
        differing_pixels -= difference_mask.histogram()[0]
    report = {
        "format": "HUNYUANOCR_REAL_OCR_ASSETS_V2",
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
            {
                "path": STB_RGB_TARGET.relative_to(PROJECT_DIR).as_posix(),
                "encoding": "production stb_image-decoded RGB PPM contract",
                "sha256": sha256(STB_RGB_TARGET),
            },
        ],
        "jpeg_decode_comparison": {
            "pillow_rgb_path": PNG_TARGET.relative_to(PROJECT_DIR).as_posix(),
            "stb_rgb_path": STB_RGB_TARGET.relative_to(PROJECT_DIR).as_posix(),
            "differing_pixels": differing_pixels,
            "total_pixels": dimensions[0] * dimensions[1],
            "channel_max_abs_difference": [value[1] for value in extrema],
            "decoder_output": completed.stdout.strip(),
        },
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(REPORT)


if __name__ == "__main__":
    main()
