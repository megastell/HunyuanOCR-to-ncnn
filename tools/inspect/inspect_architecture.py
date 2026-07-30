from __future__ import annotations

import gc
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import HunYuanVLForConditionalGeneration


MODEL_DIR = (
    Path.home()
    / "work/hunyuanocr/models/HunyuanOCR-1.5"
)

PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

TEXT_REPORT = (
    PROJECT_DIR
    / "docs/model_architecture_inventory.txt"
)

JSON_REPORT = (
    PROJECT_DIR
    / "docs/model_architecture_inventory.json"
)


def parameter_count(
    module: torch.nn.Module,
    *,
    recursive: bool,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters(
            recurse=recursive,
        )
    )


def find_module_path(
    root: torch.nn.Module,
    target: torch.nn.Module | None,
) -> str | None:
    if target is None:
        return None

    for name, module in root.named_modules():
        if module is target:
            return name if name else "<root>"

    return "<not found>"


def safe_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def build_tree(
    module: torch.nn.Module,
    *,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def walk(
        current: torch.nn.Module,
        prefix: str,
        depth: int,
    ) -> None:
        if depth >= max_depth:
            return

        for child_name, child in current.named_children():
            path = (
                f"{prefix}.{child_name}"
                if prefix
                else child_name
            )

            result.append(
                {
                    "path": path,
                    "class": child.__class__.__name__,
                    "depth": depth + 1,
                    "own_parameters": parameter_count(
                        child,
                        recursive=False,
                    ),
                    "recursive_parameters": parameter_count(
                        child,
                        recursive=True,
                    ),
                }
            )

            walk(
                child,
                path,
                depth + 1,
            )

    walk(module, "", 0)
    return result


def main() -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(9)

    print("Loading HunyuanOCR on CPU FP32...", flush=True)

    model = (
        HunYuanVLForConditionalGeneration
        .from_pretrained(
            str(MODEL_DIR),
            dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .eval()
    )

    total_parameters = parameter_count(
        model,
        recursive=True,
    )

    tree = build_tree(
        model,
        max_depth=3,
    )

    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()

    input_embedding_path = find_module_path(
        model,
        input_embedding,
    )

    output_embedding_path = find_module_path(
        model,
        output_embedding,
    )

    top_level_children: list[dict[str, Any]] = []

    for name, child in model.named_children():
        top_level_children.append(
            {
                "path": name,
                "class": child.__class__.__name__,
                "own_parameters": parameter_count(
                    child,
                    recursive=False,
                ),
                "recursive_parameters": parameter_count(
                    child,
                    recursive=True,
                ),
                "forward_signature": safe_signature(
                    child.forward,
                ),
            }
        )

    module_lists: list[dict[str, Any]] = []
    vision_candidates: list[dict[str, Any]] = []
    embedding_candidates: list[dict[str, Any]] = []
    head_candidates: list[dict[str, Any]] = []

    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        searchable = (
            name + " " + class_name
        ).lower()

        if isinstance(module, torch.nn.ModuleList):
            module_lists.append(
                {
                    "path": name or "<root>",
                    "class": class_name,
                    "length": len(module),
                    "parameters": parameter_count(
                        module,
                        recursive=True,
                    ),
                }
            )

        if any(
            token in searchable
            for token in (
                "vision",
                "visual",
                "vit",
            )
        ):
            # 限制层级，避免打印视觉模块中的每个小算子。
            if name.count(".") <= 3:
                vision_candidates.append(
                    {
                        "path": name or "<root>",
                        "class": class_name,
                        "parameters": parameter_count(
                            module,
                            recursive=True,
                        ),
                        "forward_signature": safe_signature(
                            module.forward,
                        ),
                    }
                )

        if (
            name.endswith("embed_tokens")
            or name.endswith("word_embeddings")
            or isinstance(module, torch.nn.Embedding)
        ):
            embedding_candidates.append(
                {
                    "path": name or "<root>",
                    "class": class_name,
                    "parameters": parameter_count(
                        module,
                        recursive=True,
                    ),
                }
            )

        if (
            name.endswith("lm_head")
            or name.endswith("output_projection")
            or name.endswith("output_layer")
        ):
            head_candidates.append(
                {
                    "path": name or "<root>",
                    "class": class_name,
                    "parameters": parameter_count(
                        module,
                        recursive=True,
                    ),
                    "forward_signature": safe_signature(
                        module.forward,
                    ),
                }
            )

    prefix_counts: dict[str, list[dict[str, Any]]] = {}

    parameter_names = [
        name
        for name, _ in model.named_parameters()
    ]

    for depth in (1, 2, 3):
        counter: Counter[str] = Counter()

        for name, parameter in model.named_parameters():
            parts = name.split(".")
            prefix = ".".join(parts[:depth])

            counter[prefix] += parameter.numel()

        prefix_counts[str(depth)] = [
            {
                "prefix": prefix,
                "parameters": count,
            }
            for prefix, count
            in counter.most_common(100)
        ]

    report = {
        "model_dir": str(MODEL_DIR),
        "model_class": model.__class__.__name__,
        "model_forward_signature": safe_signature(
            model.forward,
        ),
        "total_parameters": total_parameters,
        "input_embedding_path": input_embedding_path,
        "input_embedding_class": (
            input_embedding.__class__.__name__
            if input_embedding is not None
            else None
        ),
        "output_embedding_path": output_embedding_path,
        "output_embedding_class": (
            output_embedding.__class__.__name__
            if output_embedding is not None
            else None
        ),
        "top_level_children": top_level_children,
        "module_lists": module_lists,
        "vision_candidates": vision_candidates,
        "embedding_candidates": embedding_candidates,
        "head_candidates": head_candidates,
        "tree_depth_3": tree,
        "parameter_prefix_counts": prefix_counts,
        "parameter_name_count": len(parameter_names),
        "first_100_parameter_names": parameter_names[:100],
    }

    JSON_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_REPORT.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    lines: list[str] = []

    lines.append("===== MODEL =====")
    lines.append(f"class: {report['model_class']}")
    lines.append(
        f"total parameters: {total_parameters:,}"
    )
    lines.append(
        "forward: "
        + report["model_forward_signature"]
    )

    lines.append("")
    lines.append("===== INPUT / OUTPUT EMBEDDINGS =====")
    lines.append(
        "input embedding: "
        f"{input_embedding_path} "
        f"({report['input_embedding_class']})"
    )
    lines.append(
        "output embedding: "
        f"{output_embedding_path} "
        f"({report['output_embedding_class']})"
    )

    lines.append("")
    lines.append("===== TOP-LEVEL CHILDREN =====")

    for item in top_level_children:
        lines.append(
            f"{item['path']} | "
            f"{item['class']} | "
            f"params={item['recursive_parameters']:,}"
        )
        lines.append(
            f"  forward{item['forward_signature']}"
        )

    lines.append("")
    lines.append("===== MODULELIST CANDIDATES =====")

    for item in module_lists:
        lines.append(
            f"{item['path']} | "
            f"{item['class']} | "
            f"length={item['length']} | "
            f"params={item['parameters']:,}"
        )

    lines.append("")
    lines.append("===== VISION CANDIDATES =====")

    for item in vision_candidates:
        lines.append(
            f"{item['path']} | "
            f"{item['class']} | "
            f"params={item['parameters']:,}"
        )
        lines.append(
            f"  forward{item['forward_signature']}"
        )

    lines.append("")
    lines.append("===== EMBEDDING CANDIDATES =====")

    for item in embedding_candidates:
        lines.append(
            f"{item['path']} | "
            f"{item['class']} | "
            f"params={item['parameters']:,}"
        )

    lines.append("")
    lines.append("===== HEAD CANDIDATES =====")

    for item in head_candidates:
        lines.append(
            f"{item['path']} | "
            f"{item['class']} | "
            f"params={item['parameters']:,}"
        )
        lines.append(
            f"  forward{item['forward_signature']}"
        )

    lines.append("")
    lines.append("===== MODULE TREE, DEPTH <= 3 =====")

    for item in tree:
        indentation = "  " * (
            item["depth"] - 1
        )

        lines.append(
            f"{indentation}{item['path']} | "
            f"{item['class']} | "
            f"own={item['own_parameters']:,} | "
            f"recursive={item['recursive_parameters']:,}"
        )

    TEXT_REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))
    print()
    print("Text report:", TEXT_REPORT)
    print("JSON report:", JSON_REPORT)
    print("✅ 模型结构清单生成完成。")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
