from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "tests/assets/ocr_smoke_en.png"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs/ocr_smoke_image.json"
)

FONT_PATH = Path(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def main() -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(
            f"找不到固定字体：{FONT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    width = 768
    height = 320

    image = Image.new(
        mode="RGB",
        size=(width, height),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(image)

    title_font = ImageFont.truetype(
        str(FONT_PATH),
        size=68,
    )

    subtitle_font = ImageFont.truetype(
        str(FONT_PATH),
        size=52,
    )

    # 固定边框，便于观察图像是否发生裁剪或缩放异常。
    draw.rectangle(
        [(20, 20), (width - 21, height - 21)],
        outline=(0, 0, 0),
        width=3,
    )

    draw.text(
        (70, 65),
        "HELLO 2026",
        font=title_font,
        fill=(0, 0, 0),
    )

    draw.text(
        (70, 180),
        "NCNN CPU TEST",
        font=subtitle_font,
        fill=(0, 0, 0),
    )

    image.save(
        OUTPUT_PATH,
        format="PNG",
        optimize=False,
    )

    report = {
        "image_path": str(OUTPUT_PATH),
        "width": width,
        "height": height,
        "mode": image.mode,
        "font_path": str(FONT_PATH),
        "expected_text": "HELLO 2026\nNCNN CPU TEST",
        "sha256": sha256_file(OUTPUT_PATH),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("Image:", OUTPUT_PATH)
    print("SHA-256:", report["sha256"])
    print("Expected text:")
    print(report["expected_text"])
    print("✅ 固定 OCR 测试图片生成成功。")


if __name__ == "__main__":
    main()
