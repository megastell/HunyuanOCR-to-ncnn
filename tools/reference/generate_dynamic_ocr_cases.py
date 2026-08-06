from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path.home() / "work/hunyuanocr/HunyuanOCR-ncnn"
ASSET_DIR = PROJECT_DIR / "tests/assets"
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

CASES = [
    {
        "name": "ocr_wide_en",
        "size": (1024, 256),
        "lines": ["WIDE OCR", "TEST 314"],
        "font_size": 72,
        "line_spacing": 18,
    },
    {
        "name": "ocr_square_en",
        "size": (512, 512),
        "lines": ["SQUARE", "OCR TEST", "7"],
        "font_size": 64,
        "line_spacing": 20,
    },
    {
        "name": "ocr_tall_en",
        "size": (384, 768),
        "lines": ["TALL", "OCR", "TEST 42"],
        "font_size": 60,
        "line_spacing": 24,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def generate(case: dict[str, object]) -> dict[str, object]:
    name = str(case["name"])
    width, height = case["size"]
    lines = [str(line) for line in case["lines"]]
    font = ImageFont.truetype(str(FONT_PATH), int(case["font_size"]))
    spacing = int(case["line_spacing"])

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    border = max(10, min(width, height) // 32)
    draw.rectangle(
        (border, border, width - border - 1, height - border - 1),
        outline="black",
        width=max(3, border // 4),
    )

    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    total_height = sum(heights) + spacing * (len(lines) - 1)
    y = (height - total_height) // 2
    for line, box, line_height in zip(lines, boxes, heights):
        line_width = box[2] - box[0]
        x = (width - line_width) // 2
        draw.text((x, y - box[1]), line, fill="black", font=font)
        y += line_height + spacing

    path = ASSET_DIR / f"{name}.png"
    image.save(path, format="PNG", optimize=False)
    return {
        "name": name,
        "path": path.relative_to(PROJECT_DIR).as_posix(),
        "width": width,
        "height": height,
        "source_lines": lines,
        "sha256": sha256(path),
    }


def main() -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(FONT_PATH)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generated = [generate(case) for case in CASES]
    report_path = ASSET_DIR / "dynamic_ocr_cases.json"
    report_path.write_text(
        json.dumps({"format": "HUNYUANOCR_DYNAMIC_OCR_CASES_V1", "cases": generated}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    for case in generated:
        print(
            f"{case['name']}: {case['width']}x{case['height']} "
            f"sha256={case['sha256']}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
