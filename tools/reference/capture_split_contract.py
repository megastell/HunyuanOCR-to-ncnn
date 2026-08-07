from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import (
    AutoProcessor,
    HunYuanVLForConditionalGeneration,
)


PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

MODEL_DIR = (
    Path(os.environ.get(
        "HUNYUANOCR_MODEL_DIR",
        str(Path.home() / "work/hunyuanocr/models/HunyuanOCR-1.5"),
    ))
)

IMAGE_PATH = (
    Path(os.environ.get(
        "HUNYUANOCR_SMOKE_IMAGE",
        str(PROJECT_DIR / "tests/assets/ocr_smoke_en.png"),
    ))
)

REFERENCE_ROOT = (
    Path(os.environ.get(
        "HUNYUANOCR_REFERENCE_DIR",
        str(PROJECT_DIR / "reference/smoke_en_cpu_fp32"),
    ))
)

OUTPUT_DIR = (
    REFERENCE_ROOT
    / "split_contract"
)
DOCS_DIR = Path(os.environ.get("HUNYUANOCR_DOCS_DIR", str(PROJECT_DIR / "docs")))

REPORT_PATH = (
    DOCS_DIR / "split_contract_cpu_fp32.json"
)

PROMPT = (
    "请逐行识别图片中的所有文字。"
    "只输出图片中的文字本身，保留换行，不要解释。"
)

TORCH_THREADS = 9


def peak_rss_gib() -> float:
    value_kib = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    return value_kib / 1024.0 / 1024.0


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


def save_tensor(
    name: str,
    tensor: torch.Tensor,
) -> dict[str, Any]:
    tensor_cpu = tensor.detach().cpu()

    save_tensor_value = tensor_cpu

    # NumPy不能直接表示bfloat16。
    if save_tensor_value.dtype == torch.bfloat16:
        save_tensor_value = save_tensor_value.float()

    output_path = OUTPUT_DIR / f"{name}.npy"

    np.save(
        output_path,
        save_tensor_value.numpy(),
    )

    float_tensor = tensor_cpu.float()

    description = {
        "file": str(output_path),
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "numel": tensor_cpu.numel(),
        "min": float(float_tensor.min()),
        "max": float(float_tensor.max()),
        "mean": float(float_tensor.mean()),
        "sha256": sha256_file(output_path),
    }

    print(
        f"{name:32s} "
        f"shape={tuple(tensor_cpu.shape)!s:22s} "
        f"dtype={str(tensor_cpu.dtype):15s}"
    )

    return description


def first_tensor_from_output(
    output: Any,
) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        for value in output.values():
            result = first_tensor_from_output(value)

            if result is not None:
                return result

    if isinstance(output, (tuple, list)):
        for value in output:
            result = first_tensor_from_output(value)

            if result is not None:
                return result

    return None


def config_fields(
    config: Any,
    names: tuple[str, ...],
) -> dict[str, Any]:
    values: dict[str, Any] = {}

    for name in names:
        value = getattr(config, name, None)

        if isinstance(value, tuple):
            value = list(value)

        values[name] = value

    return values


def summarize_cache(
    cache: Any,
) -> dict[str, Any]:
    if cache is None:
        return {
            "class": None,
            "layer_count": 0,
        }

    report: dict[str, Any] = {
        "class": cache.__class__.__name__,
    }

    if hasattr(cache, "get_seq_length"):
        try:
            report["sequence_length"] = int(
                cache.get_seq_length()
            )
        except Exception as error:
            report["sequence_length_error"] = repr(error)

    layers = getattr(cache, "layers", None)

    if layers is None:
        report["has_layers_attribute"] = False
        return report

    report["has_layers_attribute"] = True
    report["layer_count"] = len(layers)

    layer_reports: list[dict[str, Any]] = []

    possible_names = (
        "keys",
        "values",
        "key",
        "value",
        "key_states",
        "value_states",
    )

    for layer_index, layer in enumerate(layers):
        layer_report: dict[str, Any] = {
            "index": layer_index,
            "class": layer.__class__.__name__,
        }

        tensors: dict[str, Any] = {}

        for attribute_name in possible_names:
            value = getattr(
                layer,
                attribute_name,
                None,
            )

            if isinstance(value, torch.Tensor):
                tensors[attribute_name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "numel": value.numel(),
                }

        # 若属性名称不同，检查对象自身字典中的张量。
        if not tensors and hasattr(layer, "__dict__"):
            for attribute_name, value in layer.__dict__.items():
                if isinstance(value, torch.Tensor):
                    tensors[attribute_name] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "numel": value.numel(),
                    }

        layer_report["tensors"] = tensors
        layer_reports.append(layer_report)

    report["layers"] = layer_reports

    return report


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    print("===== Load processor =====", flush=True)

    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        backend="pil",
        local_files_only=True,
    )

    print(
        "Processor:",
        processor.__class__.__name__,
        flush=True,
    )

    print("\n===== Load model =====", flush=True)

    load_start = time.perf_counter()

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

    load_seconds = time.perf_counter() - load_start

    print(
        "Model:",
        model.__class__.__name__,
        flush=True,
    )
    print(
        "Load time:",
        f"{load_seconds:.3f} s",
        flush=True,
    )

    text_config = model.config.text_config
    vision_config = model.config.vision_config

    text_summary = config_fields(
        text_config,
        (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
        ),
    )

    vision_summary = config_fields(
        vision_config,
        (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "depth",
            "num_heads",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "out_hidden_size",
            "num_position_embeddings",
        ),
    )

    print("\n===== Text configuration =====")

    for name, value in text_summary.items():
        print(f"{name:28s}: {value}")

    print("\n===== Vision configuration =====")

    for name, value in vision_summary.items():
        print(f"{name:28s}: {value}")

    embed_tokens = (
        model.model
        .language_model
        .embed_tokens
    )

    decoder_layers = (
        model.model
        .language_model
        .layers
    )

    final_norm = (
        model.model
        .language_model
        .norm
    )

    vision_tower = (
        model.model
        .vision_tower
    )

    lm_head = model.lm_head

    embed_weight = embed_tokens.weight
    lm_head_weight = lm_head.weight

    same_python_object = (
        embed_weight is lm_head_weight
    )

    same_storage = (
        embed_weight.untyped_storage().data_ptr()
        ==
        lm_head_weight.untyped_storage().data_ptr()
    )

    print("\n===== Weight sharing =====")
    print(
        "Embedding weight shape:",
        tuple(embed_weight.shape),
    )
    print(
        "LM head weight shape:",
        tuple(lm_head_weight.shape),
    )
    print(
        "Same Python object:",
        same_python_object,
    )
    print(
        "Same storage:",
        same_storage,
    )
    print(
        "Decoder layer count:",
        len(decoder_layers),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "path": str(IMAGE_PATH),
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    captures: dict[str, torch.Tensor] = {}

    def embedding_hook(
        module: torch.nn.Module,
        module_inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, module_inputs

        if isinstance(output, torch.Tensor):
            captures["text_embedding_output"] = (
                output.detach()
            )

    def vision_hook(
        module: torch.nn.Module,
        module_inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, module_inputs

        last_hidden_state = getattr(
            output,
            "last_hidden_state",
            None,
        )

        pooler_output = getattr(
            output,
            "pooler_output",
            None,
        )

        if isinstance(
            last_hidden_state,
            torch.Tensor,
        ):
            captures[
                "vision_last_hidden_state"
            ] = last_hidden_state.detach()

        if isinstance(
            pooler_output,
            torch.Tensor,
        ):
            captures[
                "vision_pooler_output"
            ] = pooler_output.detach()

        if (
            "vision_last_hidden_state"
            not in captures
            and
            "vision_pooler_output"
            not in captures
        ):
            first_tensor = first_tensor_from_output(
                output
            )

            if first_tensor is not None:
                captures[
                    "vision_first_tensor"
                ] = first_tensor.detach()

    def norm_hook(
        module: torch.nn.Module,
        module_inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, module_inputs

        if isinstance(output, torch.Tensor):
            captures[
                "final_norm_output"
            ] = output.detach()

    def lm_head_hook(
        module: torch.nn.Module,
        module_inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module

        if (
            module_inputs
            and isinstance(
                module_inputs[0],
                torch.Tensor,
            )
        ):
            captures[
                "lm_head_input"
            ] = module_inputs[0].detach()

        if isinstance(output, torch.Tensor):
            captures[
                "lm_head_output"
            ] = output.detach()

    handles = [
        embed_tokens.register_forward_hook(
            embedding_hook
        ),
        vision_tower.register_forward_hook(
            vision_hook
        ),
        final_norm.register_forward_hook(
            norm_hook
        ),
        lm_head.register_forward_hook(
            lm_head_hook
        ),
    ]

    print("\n===== Prefill forward =====", flush=True)

    forward_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )

    forward_seconds = (
        time.perf_counter() - forward_start
    )

    for handle in handles:
        handle.remove()

    if outputs.hidden_states:
        captures[
            "model_last_hidden_state"
        ] = outputs.hidden_states[-1].detach()

    captures["forward_logits"] = (
        outputs.logits.detach()
    )

    print(
        "Forward time:",
        f"{forward_seconds:.3f} s",
    )
    print(
        "Peak RSS:",
        f"{peak_rss_gib():.3f} GiB",
    )

    print("\n===== Captured tensors =====")

    tensor_reports: dict[str, Any] = {}

    for name in sorted(captures):
        tensor_reports[name] = save_tensor(
            name,
            captures[name],
        )

    generated_tokens_path = (
        REFERENCE_ROOT
        / "generated_token_ids.json"
    )

    reference_token_ids = json.loads(
        generated_tokens_path.read_text(
            encoding="utf-8"
        )
    )

    expected_first_token = int(
        reference_token_ids[0]
    )

    actual_first_token = int(
        outputs.logits[
            0,
            -1,
        ]
        .argmax(dim=-1)
        .item()
    )

    print("\n===== First-token validation =====")
    print(
        "Expected first token:",
        expected_first_token,
    )
    print(
        "Actual first token:",
        actual_first_token,
    )
    print(
        "First token matches:",
        actual_first_token
        == expected_first_token,
    )

    if actual_first_token != expected_first_token:
        raise RuntimeError(
            "普通forward得到的首token"
            "与generate参考结果不一致。"
        )

    cache_report = summarize_cache(
        outputs.past_key_values
    )

    print("\n===== KV cache =====")
    print(
        "Cache class:",
        cache_report.get("class"),
    )
    print(
        "Cache sequence length:",
        cache_report.get(
            "sequence_length"
        ),
    )
    print(
        "Cache layer count:",
        cache_report.get(
            "layer_count"
        ),
    )

    for layer in cache_report.get(
        "layers",
        [],
    ):
        print(
            f"Layer {layer['index']:02d}: "
            f"{layer['tensors']}"
        )

    report = {
        "model_revision": (
            PROJECT_DIR
            / "configs/model_revision.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),
        "model_class": model.__class__.__name__,
        "processor_class": (
            processor.__class__.__name__
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": (
            transformers.__version__
        ),
        "device": "cpu",
        "dtype": "float32",
        "attention_implementation": "eager",
        "torch_threads": TORCH_THREADS,
        "model_load_seconds": load_seconds,
        "prefill_forward_seconds": forward_seconds,
        "peak_rss_gib": peak_rss_gib(),
        "text_config": text_summary,
        "vision_config": vision_summary,
        "module_paths": {
            "vision_tower": (
                "model.vision_tower"
            ),
            "text_embedding": (
                "model.language_model.embed_tokens"
            ),
            "decoder_layers": (
                "model.language_model.layers"
            ),
            "final_norm": (
                "model.language_model.norm"
            ),
            "lm_head": "lm_head",
        },
        "weight_sharing": {
            "embedding_weight_shape": list(
                embed_weight.shape
            ),
            "lm_head_weight_shape": list(
                lm_head_weight.shape
            ),
            "same_python_object": (
                same_python_object
            ),
            "same_storage": same_storage,
        },
        "decoder_layer_count": len(
            decoder_layers
        ),
        "input_shapes": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in inputs.items()
            if isinstance(value, torch.Tensor)
        },
        "captured_tensors": tensor_reports,
        "expected_first_token": (
            expected_first_token
        ),
        "actual_first_token": (
            actual_first_token
        ),
        "first_token_matches": (
            actual_first_token
            == expected_first_token
        ),
        "kv_cache": cache_report,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\nReport:", REPORT_PATH)
    print("Tensor directory:", OUTPUT_DIR)
    print(
        "✅ HunyuanOCR模型拆分契约捕获成功。"
    )


if __name__ == "__main__":
    main()
