from __future__ import annotations

from pathlib import Path


PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

PARAM_PATH = (
    PROJECT_DIR
    / "artifacts/decoder_layer0_prefill"
    / "decoder_layer0_prefill.ncnn.param"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "docs/decoder_layer0_ncnn_layers.txt"
)


def main() -> None:
    lines = PARAM_PATH.read_text(
        encoding="utf-8"
    ).splitlines()

    if len(lines) < 3:
        raise RuntimeError("ncnn param内容不完整。")

    output: list[str] = [
        f"magic: {lines[0]}",
        f"header: {lines[1]}",
        "",
    ]

    for index, line in enumerate(
        lines[2:],
        start=0,
    ):
        parts = line.split()

        if len(parts) < 4:
            output.append(
                f"{index:02d} INVALID: {line}"
            )
            continue

        layer_type = parts[0]
        layer_name = parts[1]
        bottom_count = int(parts[2])
        top_count = int(parts[3])

        cursor = 4
        bottoms = parts[
            cursor:cursor + bottom_count
        ]
        cursor += bottom_count

        tops = parts[
            cursor:cursor + top_count
        ]
        cursor += top_count

        parameters = parts[cursor:]

        output.append(
            f"{index:02d} "
            f"{layer_type:18s} "
            f"{layer_name:24s} "
            f"bottoms={','.join(bottoms) or '-':16s} "
            f"tops={','.join(tops) or '-':16s} "
            f"params={' '.join(parameters)}"
        )

    OUTPUT_PATH.write_text(
        "\n".join(output) + "\n",
        encoding="utf-8",
    )

    print("\n".join(output))
    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
