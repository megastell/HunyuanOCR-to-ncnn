#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <net.h>

struct BoundaryMetrics {
    double maximum_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double cosine_similarity = 0.0;
};

struct MultimodalPrefillInput {
    std::vector<float> hidden_states;
    std::array<std::int64_t, 3> image_grid_thw = {0, 0, 0};
    int original_width = 0;
    int original_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int image_token_start = 0;
    int image_token_end = 0;
    BoundaryMetrics original_rgb_metrics;
    BoundaryMetrics resized_rgb_metrics;
    BoundaryMetrics pixel_values_metrics;
    BoundaryMetrics vision_embedding_metrics;
    BoundaryMetrics text_embedding_metrics;
    BoundaryMetrics fused_hidden_metrics;
};

bool build_multimodal_prefill_input(
    const std::string& project_root,
    const std::string& image_path,
    bool use_packing_layout,
    ncnn::Net& text_embedding_network,
    MultimodalPrefillInput& result);
