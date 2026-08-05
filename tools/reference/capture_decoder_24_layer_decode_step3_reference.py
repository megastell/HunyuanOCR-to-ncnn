from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from transformers import HunYuanVLForConditionalGeneration


PROJECT_DIR = (
    Path.home()
    / "work/hunyuanocr/HunyuanOCR-ncnn"
)

MODEL_DIR = (
    Path.home()
    / "work/hunyuanocr/models/HunyuanOCR-1.5"
)

BASE_CAPTURE_SCRIPT = (
    PROJECT_DIR
    / "tools/reference"
    / "capture_decoder_layer0_decode_reference.py"
)

GENERATED_TOKENS_PATH = (
    PROJECT_DIR
    / "docs"
    / "reference_smoke_cpu_fp32.json"
)

REPORT_PATH = (
    PROJECT_DIR
    / "docs"
    / "decoder_24_layer_decode_step3_reference.json"
)

TORCH_THREADS = 9
LAYER_COUNT = 24

KV_HEADS = 8
HEAD_DIM = 128

PREFILL_LENGTH = 313
STEP1_PRESENT_LENGTH = 314
STEP2_PRESENT_LENGTH = 315
STEP3_PRESENT_LENGTH = 316


def load_base_module() -> Any:
    specification = (
        importlib.util.spec_from_file_location(
            "decoder_decode_reference_base",
            BASE_CAPTURE_SCRIPT,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            "无法加载现有Decode参考捕获模块。"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


def dtype_tag(value: torch.Tensor) -> str:
    mapping = {
        torch.float32: "f32",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.int64: "i64",
        torch.int32: "i32",
        torch.bool: "bool",
    }

    return mapping.get(
        value.dtype,
        str(value.dtype).replace("torch.", ""),
    )


def tensor_summary(
    value: torch.Tensor,
) -> dict[str, Any]:
    detached = (
        value.detach()
        .cpu()
        .contiguous()
    )

    numeric = detached.float()
    finite = numeric[torch.isfinite(numeric)]

    if finite.numel() > 0:
        minimum = float(finite.min().item())
        maximum = float(finite.max().item())
        mean = float(finite.mean().item())
    else:
        minimum = None
        maximum = None
        mean = None

    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "minimum_finite": minimum,
        "maximum_finite": maximum,
        "mean_finite": mean,
    }


def layer_reference_dir(
    layer_index: int,
) -> Path:
    return (
        PROJECT_DIR
        / "reference/smoke_en_cpu_fp32"
        / f"decoder_layer{layer_index}_decode_step3"
    )


def layer_raw_dir(
    layer_index: int,
) -> Path:
    return (
        PROJECT_DIR
        / f"artifacts/decoder_layer{layer_index}_decode_step3"
        / "reference"
    )


def save_tensor(
    layer_index: int,
    name: str,
    value: torch.Tensor,
) -> dict[str, Any]:
    detached = (
        value.detach()
        .cpu()
        .contiguous()
    )

    npy_directory = layer_reference_dir(
        layer_index
    )

    raw_directory = layer_raw_dir(
        layer_index
    )

    npy_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    npy_path = (
        npy_directory
        / f"{name}.npy"
    )

    tag = dtype_tag(detached)

    raw_path = (
        raw_directory
        / f"{name}_{tag}.bin"
    )

    array = detached.numpy()

    np.save(
        npy_path,
        array,
    )

    array.tofile(
        raw_path
    )

    report = tensor_summary(
        detached
    )

    report.update(
        {
            "npy_path": (
                npy_path.relative_to(
                    PROJECT_DIR
                ).as_posix()
            ),
            "raw_path": (
                raw_path.relative_to(
                    PROJECT_DIR
                ).as_posix()
            ),
            "raw_bytes": (
                raw_path.stat().st_size
            ),
        }
    )

    return report


def main() -> None:
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_num_threads(TORCH_THREADS)

    base = load_base_module()
    contract = base.load_contract_module()

    model_inputs = {
        name: contract.load_tensor(name)
        for name in (
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        )
    }

    generation_report = json.loads(
        GENERATED_TOKENS_PATH.read_text(
            encoding="utf-8"
        )
    )

    generated_tokens = generation_report[
        "generated_token_ids"
    ]

    if len(generated_tokens) < 4:
        raise RuntimeError(
            "参考生成序列少于4个token。"
        )

    expected_prefill_token = int(
        generated_tokens[0]
    )

    expected_step1_token = int(
        generated_tokens[1]
    )

    expected_step2_token = int(
        generated_tokens[2]
    )

    expected_step3_token = int(
        generated_tokens[3]
    )

    print("===== Token contract =====")
    print(
        "Prefill output token:",
        expected_prefill_token,
    )
    print(
        "Step 1 output token:",
        expected_step1_token,
    )
    print(
        "Step 2 output token:",
        expected_step2_token,
    )
    print(
        "Step 3 output token:",
        expected_step3_token,
    )

    if (
        expected_prefill_token,
        expected_step1_token,
        expected_step2_token,
        expected_step3_token,
    ) != (
        93892,
        5112,
        206,
        1717,
    ):
        raise RuntimeError(
            "前四个Token契约不符合预期。"
        )

    print()
    print("===== Load model once =====")

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

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    decoder_layers = (
        model.model
        .language_model
        .layers
    )

    if len(decoder_layers) != LAYER_COUNT:
        raise RuntimeError(
            "Decoder层数量错误："
            f"实际={len(decoder_layers)}，"
            f"预期={LAYER_COUNT}"
        )

    print(
        f"Model loaded in {load_seconds:.3f}s"
    )

    print()
    print("===== Prefill =====")

    with torch.inference_mode():
        prefill_output = model(
            **model_inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

    prefill_token = (
        prefill_output.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    if int(prefill_token.item()) != (
        expected_prefill_token
    ):
        raise RuntimeError(
            "Prefill输出token与参考不一致。"
        )

    full_input_ids_step1 = torch.cat(
        (
            model_inputs["input_ids"],
            prefill_token,
        ),
        dim=1,
    )

    attention_mask_step1 = torch.cat(
        (
            model_inputs["attention_mask"],
            torch.ones_like(prefill_token),
        ),
        dim=1,
    )

    mm_types_step1 = torch.cat(
        (
            model_inputs["mm_token_type_ids"],
            model_inputs[
                "mm_token_type_ids"
            ].new_zeros(
                (
                    model_inputs[
                        "mm_token_type_ids"
                    ].shape[0],
                    1,
                )
            ),
        ),
        dim=1,
    )

    with torch.inference_mode():
        prepared_step1 = (
            model.prepare_inputs_for_generation(
                input_ids=full_input_ids_step1,
                next_sequence_length=1,
                past_key_values=(
                    prefill_output.past_key_values
                ),
                attention_mask=attention_mask_step1,
                mm_token_type_ids=mm_types_step1,
                pixel_values=model_inputs[
                    "pixel_values"
                ],
                image_grid_thw=model_inputs[
                    "image_grid_thw"
                ],
                use_cache=True,
            )
        )

        step1_arguments = dict(
            prepared_step1
        )

        step1_arguments.update(
            {
                "use_cache": True,
                "return_dict": True,
                "logits_to_keep": 1,
            }
        )

        step1_output = model(
            **step1_arguments
        )

    step1_token = (
        step1_output.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    if int(step1_token.item()) != (
        expected_step1_token
    ):
        raise RuntimeError(
            "Step 1输出token与参考不一致。"
        )

    step1_legacy = contract.cache_to_legacy(
        step1_output.past_key_values
    )

    if len(step1_legacy) != LAYER_COUNT:
        raise RuntimeError(
            "Step 1 KV层数量错误。"
        )

    full_input_ids_step2 = torch.cat(
        (
            full_input_ids_step1,
            step1_token,
        ),
        dim=1,
    )

    attention_mask_step2 = torch.cat(
        (
            attention_mask_step1,
            torch.ones_like(step1_token),
        ),
        dim=1,
    )

    mm_types_step2 = torch.cat(
        (
            mm_types_step1,
            mm_types_step1.new_zeros(
                (
                    mm_types_step1.shape[0],
                    1,
                )
            ),
        ),
        dim=1,
    )

    with torch.inference_mode():
        prepared_step2 = (
            model.prepare_inputs_for_generation(
                input_ids=full_input_ids_step2,
                next_sequence_length=1,
                past_key_values=(
                    step1_output.past_key_values
                ),
                attention_mask=attention_mask_step2,
                mm_token_type_ids=mm_types_step2,
                pixel_values=model_inputs[
                    "pixel_values"
                ],
                image_grid_thw=model_inputs[
                    "image_grid_thw"
                ],
                use_cache=True,
            )
        )

    if tuple(
        prepared_step2["input_ids"].shape
    ) != (1, 1):
        raise RuntimeError(
            "Step 2 input_ids形状错误。"
        )

    if int(
        prepared_step2["input_ids"].item()
    ) != expected_step1_token:
        raise RuntimeError(
            "Step 2输入token不是5112。"
        )

    # 先正常执行Step 2，不注册Hook。
    # Step 2的输出KV长度为315，
    # 它将成为Step 3的past KV。
    step2_arguments = dict(
        prepared_step2
    )

    step2_arguments.update(
        {
            "use_cache": True,
            "return_dict": True,
            "logits_to_keep": 1,
        }
    )

    with torch.inference_mode():
        step2_output = model(
            **step2_arguments
        )

    step2_token = (
        step2_output.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    if int(step2_token.item()) != (
        expected_step2_token
    ):
        raise RuntimeError(
            "Step 2输出token与参考不一致。"
        )

    step2_legacy = contract.cache_to_legacy(
        step2_output.past_key_values
    )

    if len(step2_legacy) != LAYER_COUNT:
        raise RuntimeError(
            "Step 2 KV层数量错误。"
        )

    full_input_ids_step3 = torch.cat(
        (
            full_input_ids_step2,
            step2_token,
        ),
        dim=1,
    )

    attention_mask_step3 = torch.cat(
        (
            attention_mask_step2,
            torch.ones_like(
                step2_token
            ),
        ),
        dim=1,
    )

    mm_types_step3 = torch.cat(
        (
            mm_types_step2,
            mm_types_step2.new_zeros(
                (
                    mm_types_step2.shape[0],
                    1,
                )
            ),
        ),
        dim=1,
    )

    with torch.inference_mode():
        prepared_step3 = (
            model.prepare_inputs_for_generation(
                input_ids=full_input_ids_step3,
                next_sequence_length=1,
                past_key_values=(
                    step2_output.past_key_values
                ),
                attention_mask=attention_mask_step3,
                mm_token_type_ids=mm_types_step3,
                pixel_values=model_inputs[
                    "pixel_values"
                ],
                image_grid_thw=model_inputs[
                    "image_grid_thw"
                ],
                use_cache=True,
            )
        )

    if tuple(
        prepared_step3["input_ids"].shape
    ) != (1, 1):
        raise RuntimeError(
            "Step 3 input_ids形状错误。"
        )

    if int(
        prepared_step3["input_ids"].item()
    ) != expected_step2_token:
        raise RuntimeError(
            "Step 3输入token不是206。"
        )

    if tuple(
        prepared_step3[
            "attention_mask"
        ].shape
    )[-1] != STEP3_PRESENT_LENGTH:
        raise RuntimeError(
            "Step 3 attention mask长度不是316："
            f"{tuple(prepared_step3['attention_mask'].shape)}"
        )

    captures: dict[
        str,
        torch.Tensor,
    ] = {}

    handles: list[Any] = []

    def make_layer_pre_hook(
        layer_index: int,
    ) -> Callable[..., None]:
        prefix = f"layer{layer_index}"

        def hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            arguments = (
                base.bind_forward_arguments(
                    module,
                    args,
                    kwargs,
                )
            )

            for name in (
                "hidden_states",
                "attention_mask",
                "position_embeddings",
            ):
                if name not in arguments:
                    raise RuntimeError(
                        f"Layer {layer_index}缺少参数："
                        f"{name}"
                    )

                base.clone_nested_tensors(
                    f"{prefix}_{name}",
                    arguments[name],
                    captures,
                )

        return hook

    def make_layer_post_hook(
        layer_index: int,
    ) -> Callable[..., None]:
        prefix = f"layer{layer_index}"

        def hook(
            module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module, args, kwargs

            base.clone_nested_tensors(
                f"{prefix}_output",
                output,
                captures,
            )

        return hook

    for layer_index, layer in enumerate(
        decoder_layers
    ):
        handles.append(
            layer.register_forward_pre_hook(
                make_layer_pre_hook(
                    layer_index
                ),
                with_kwargs=True,
            )
        )

        handles.append(
            layer.register_forward_hook(
                make_layer_post_hook(
                    layer_index
                ),
                with_kwargs=True,
            )
        )

    print()
    print(
        "===== Capture Step 3 for all 24 layers ====="
    )

    decode_start = time.perf_counter()

    step3_arguments = dict(
        prepared_step3
    )

    step3_arguments.update(
        {
            "use_cache": True,
            "return_dict": True,
            "logits_to_keep": 1,
        }
    )

    try:
        with torch.inference_mode():
            step3_output = model(
                **step3_arguments
            )
    finally:
        for handle in handles:
            handle.remove()

    decode_seconds = (
        time.perf_counter()
        - decode_start
    )

    step3_token = (
        step3_output.logits[:, -1, :]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )

    if int(step3_token.item()) != (
        expected_step3_token
    ):
        raise RuntimeError(
            "Step 3输出token与参考不一致。"
        )

    step3_legacy = contract.cache_to_legacy(
        step3_output.past_key_values
    )

    if len(step3_legacy) != LAYER_COUNT:
        raise RuntimeError(
            "Step 3 KV层数量错误。"
        )

    shared_tensors = {
        "prepared_input_ids": (
            prepared_step3["input_ids"]
            .detach().cpu().clone()
        ),
        "prepared_attention_mask": (
            prepared_step3["attention_mask"]
            .detach().cpu().clone()
        ),
        "prepared_mm_token_type_ids": (
            prepared_step3[
                "mm_token_type_ids"
            ].detach().cpu().clone()
        ),
        "step3_input_token_id": (
            step2_token.detach()
            .cpu().clone()
        ),
        "next_token_id": (
            step3_token.detach()
            .cpu().clone()
        ),
        "decode_logits": (
            step3_output.logits
            .detach().cpu().clone()
        ),
    }

    expected_past_shape = (
        1,
        KV_HEADS,
        STEP2_PRESENT_LENGTH,
        HEAD_DIM,
    )

    expected_present_shape = (
        1,
        KV_HEADS,
        STEP3_PRESENT_LENGTH,
        HEAD_DIM,
    )

    expected_step3_hidden = (
        model.get_input_embeddings()(
            step2_token
        )
        .detach()
        .cpu()
        .contiguous()
    )

    captured_step3_hidden = captures[
        "layer0_hidden_states"
    ].detach().cpu().contiguous()

    embedding_hidden_error = float(
        (
            captured_step3_hidden
            - expected_step3_hidden
        )
        .abs()
        .max()
        .item()
    )

    if embedding_hidden_error != 0.0:
        raise RuntimeError(
            "Step 3 Layer 0 hidden与"
            "token 206 Embedding不一致。"
        )

    maximum_hidden_boundary_error = 0.0

    for layer_index in range(
        LAYER_COUNT - 1
    ):
        current_output = captures[
            f"layer{layer_index}_output"
        ]

        next_input = captures[
            f"layer{layer_index + 1}"
            "_hidden_states"
        ]

        boundary_error = float(
            (
                current_output
                - next_input
            )
            .abs()
            .max()
            .item()
        )

        maximum_hidden_boundary_error = max(
            maximum_hidden_boundary_error,
            boundary_error,
        )

        if boundary_error != 0.0:
            raise RuntimeError(
                "Decoder hidden层间边界变化："
                f"{layer_index} -> "
                f"{layer_index + 1}，"
                f"error={boundary_error}"
            )

    report_layers: list[
        dict[str, Any]
    ] = []

    maximum_key_prefix_error = 0.0
    maximum_value_prefix_error = 0.0

    print()
    print("===== Per-layer validation =====")
    print(
        "layer  key_prefix_error       "
        "value_prefix_error"
    )

    for layer_index in range(
        LAYER_COUNT
    ):
        prefix = f"layer{layer_index}"

        required_names = (
            f"{prefix}_hidden_states",
            f"{prefix}_attention_mask",
            f"{prefix}_position_embeddings_0",
            f"{prefix}_position_embeddings_1",
            f"{prefix}_output",
        )

        for name in required_names:
            if name not in captures:
                raise RuntimeError(
                    f"缺少捕获张量：{name}"
                )

        past_key = (
            step2_legacy[layer_index][0]
            .detach().cpu().clone()
        )

        past_value = (
            step2_legacy[layer_index][1]
            .detach().cpu().clone()
        )

        present_key = (
            step3_legacy[layer_index][0]
            .detach().cpu().clone()
        )

        present_value = (
            step3_legacy[layer_index][1]
            .detach().cpu().clone()
        )

        if tuple(past_key.shape) != (
            expected_past_shape
        ):
            raise RuntimeError(
                f"Layer {layer_index} past key形状错误："
                f"{tuple(past_key.shape)}"
            )

        if tuple(present_key.shape) != (
            expected_present_shape
        ):
            raise RuntimeError(
                f"Layer {layer_index} present key形状错误："
                f"{tuple(present_key.shape)}"
            )

        key_prefix_error = float(
            (
                present_key[
                    :,
                    :,
                    :STEP2_PRESENT_LENGTH,
                    :,
                ]
                - past_key
            ).abs().max().item()
        )

        value_prefix_error = float(
            (
                present_value[
                    :,
                    :,
                    :STEP2_PRESENT_LENGTH,
                    :,
                ]
                - past_value
            ).abs().max().item()
        )

        if key_prefix_error != 0.0:
            raise RuntimeError(
                f"Layer {layer_index} Key历史前缀变化。"
            )

        if value_prefix_error != 0.0:
            raise RuntimeError(
                f"Layer {layer_index} Value历史前缀变化。"
            )

        maximum_key_prefix_error = max(
            maximum_key_prefix_error,
            key_prefix_error,
        )

        maximum_value_prefix_error = max(
            maximum_value_prefix_error,
            value_prefix_error,
        )

        layer_tensors = {
            f"{prefix}_hidden_states":
                captures[
                    f"{prefix}_hidden_states"
                ],

            f"{prefix}_attention_mask":
                captures[
                    f"{prefix}_attention_mask"
                ],

            f"{prefix}_position_embeddings_0":
                captures[
                    f"{prefix}_position_embeddings_0"
                ],

            f"{prefix}_position_embeddings_1":
                captures[
                    f"{prefix}_position_embeddings_1"
                ],

            f"{prefix}_output":
                captures[
                    f"{prefix}_output"
                ],

            "past_key": past_key,
            "past_value": past_value,
            "present_key": present_key,
            "present_value": present_value,
            **shared_tensors,
        }

        tensor_reports: dict[
            str,
            Any,
        ] = {}

        for name, value in sorted(
            layer_tensors.items()
        ):
            tensor_reports[name] = save_tensor(
                layer_index,
                name,
                value,
            )

        report_layers.append(
            {
                "layer_index": layer_index,
                "past_key_shape":
                    list(past_key.shape),
                "past_value_shape":
                    list(past_value.shape),
                "present_key_shape":
                    list(present_key.shape),
                "present_value_shape":
                    list(present_value.shape),
                "key_prefix_max_abs_error":
                    key_prefix_error,
                "value_prefix_max_abs_error":
                    value_prefix_error,
                "tensors": tensor_reports,
            }
        )

        print(
            f"{layer_index:5d}  "
            f"{key_prefix_error:.10e}  "
            f"{value_prefix_error:.10e}"
        )

    report = {
        "model_revision": (
            PROJECT_DIR
            / "configs/model_revision.txt"
        ).read_text(
            encoding="utf-8"
        ).strip(),
        "torch_version": torch.__version__,
        "torch_threads": TORCH_THREADS,
        "layer_count": LAYER_COUNT,
        "model_load_seconds": load_seconds,
        "step3_decode_seconds": decode_seconds,
        "token_contract": {
            "prefill_output_token":
                expected_prefill_token,
            "step1_output_token":
                expected_step1_token,
            "step2_output_token":
                expected_step2_token,
            "step3_output_token":
                expected_step3_token,
        },
        "cache_contract": {
            "prefill_length":
                PREFILL_LENGTH,
            "step1_present_length":
                STEP1_PRESENT_LENGTH,
            "step2_present_length":
                STEP2_PRESENT_LENGTH,
            "step3_present_length":
                STEP3_PRESENT_LENGTH,
            "maximum_key_prefix_error":
                maximum_key_prefix_error,
            "maximum_value_prefix_error":
                maximum_value_prefix_error,
            "embedding_hidden_max_abs_error":
                embedding_hidden_error,
            "maximum_hidden_boundary_error":
                maximum_hidden_boundary_error,
        },
        "layers": report_layers,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Maximum key prefix error:",
        maximum_key_prefix_error,
    )
    print(
        "Maximum value prefix error:",
        maximum_value_prefix_error,
    )
    print(
        "Embedding hidden error:",
        embedding_hidden_error,
    )
    print(
        "Maximum hidden boundary error:",
        maximum_hidden_boundary_error,
    )
    print(
        "Step 3 input token:",
        int(step2_token.item()),
    )
    print(
        "Step 3 output token:",
        int(step3_token.item()),
    )
    print("Report:", REPORT_PATH)

    print(
        "✅ 全部24层Step 3参考在一次模型运行中捕获成功。"
    )


if __name__ == "__main__":
    main()
