#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct PromptInputs {
    std::vector<std::int64_t> input_ids;
    std::vector<std::int64_t> attention_mask;
    std::vector<std::int64_t> mm_token_type_ids;
    std::vector<std::int64_t> position_ids;
    std::vector<float> causal_mask;
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    int image_token_start = 0;
    int image_token_end = 0;
};

bool build_fixed_ocr_prompt_inputs(
    const std::string& project_root,
    const std::array<std::int64_t, 3>& image_grid_thw,
    PromptInputs& result);

bool build_decode_position_inputs(
    int sequence_position,
    int present_length,
    std::vector<float>& attention_mask,
    std::vector<float>& rope_cos,
    std::vector<float>& rope_sin);
