#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <net.h>

#include "detail/prompt_inputs.h"

struct MultimodalPrefillInput {
    std::vector<float> hidden_states;
    PromptInputs prompt_inputs;
    std::array<std::int64_t, 3> image_grid_thw = {0, 0, 0};
    int original_width = 0;
    int original_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int image_token_start = 0;
    int image_token_end = 0;
};

struct MultimodalResources {
    std::vector<float> vision_position_embedding;
    std::vector<float> vision_merger_constants;
    std::vector<std::int64_t> prompt_token_ids;
};

bool load_multimodal_resources(
    const std::string& model_directory,
    MultimodalResources& resources,
    std::string& error);

bool build_multimodal_prefill_input(
    const std::string& model_directory,
    const std::string& image_path,
    bool use_packing_layout,
    int num_threads,
    int max_vision_patches,
    ncnn::Net& text_embedding_network,
    const MultimodalResources& resources,
    std::string& error,
    MultimodalPrefillInput& result);
