#include "prompt_inputs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename T>
bool load_binary(
    const std::string& path,
    std::size_t expected_count,
    std::vector<T>& values)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()
        || file.tellg() != static_cast<std::streamsize>(
            expected_count * sizeof(T))) {
        std::cerr << "Invalid reference file: " << path << '\n';
        return false;
    }
    file.seekg(0, std::ios::beg);
    values.resize(expected_count);
    return static_cast<bool>(file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T))));
}

double maximum_difference(
    const std::vector<float>& actual,
    const std::vector<float>& expected)
{
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(static_cast<double>(actual[index]) - expected[index]));
    }
    return maximum;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <project_root>\n";
        return EXIT_FAILURE;
    }
    const std::string root = argv[1];
    PromptInputs actual;
    if (!build_fixed_ocr_prompt_inputs(root, {1, 22, 50}, actual)) {
        return EXIT_FAILURE;
    }

    const std::string multimodal =
        root + "/artifacts/multimodal_prefill_input/reference";
    const std::string prefill =
        root + "/artifacts/decoder_layer0_prefill_kv/reference";
    std::vector<std::int64_t> expected_ids;
    std::vector<std::int64_t> expected_attention;
    std::vector<std::int64_t> expected_mm_types;
    std::vector<std::int64_t> expected_positions;
    std::vector<float> expected_mask;
    std::vector<float> expected_cos;
    std::vector<float> expected_sin;
    if (!load_binary(multimodal + "/input_ids_i64.bin", 313, expected_ids)
        || !load_binary(
            multimodal + "/attention_mask_i64.bin", 313, expected_attention)
        || !load_binary(
            multimodal + "/mm_token_type_ids_i64.bin", 313, expected_mm_types)
        || !load_binary(
            multimodal + "/position_ids_i64.bin", 4 * 313, expected_positions)
        || !load_binary(
            prefill + "/layer0_attention_mask_f32.bin",
            313 * 313,
            expected_mask)
        || !load_binary(
            prefill + "/layer0_position_embeddings_0_f32.bin",
            4 * 313 * 128,
            expected_cos)
        || !load_binary(
            prefill + "/layer0_position_embeddings_1_f32.bin",
            4 * 313 * 128,
            expected_sin)) {
        return EXIT_FAILURE;
    }

    const bool ids_exact = actual.input_ids == expected_ids;
    const bool attention_exact = actual.attention_mask == expected_attention;
    const bool mm_types_exact = actual.mm_token_type_ids == expected_mm_types;
    const bool positions_exact = actual.position_ids == expected_positions;
    const bool mask_exact = actual.causal_mask == expected_mask;
    const double cos_max = maximum_difference(actual.rope_cos, expected_cos);
    const double sin_max = maximum_difference(actual.rope_sin, expected_sin);

    bool decode_masks_exact = true;
    double decode_cos_max = 0.0;
    double decode_sin_max = 0.0;
    for (int step = 1; step <= 10; ++step) {
        const int position = 313 + step - 1;
        const int present_length = position + 1;
        std::vector<float> decode_mask;
        std::vector<float> decode_cos;
        std::vector<float> decode_sin;
        if (!build_decode_position_inputs(
                position,
                present_length,
                decode_mask,
                decode_cos,
                decode_sin)) {
            return EXIT_FAILURE;
        }
        std::string name = "decoder_layer0_decode";
        if (step > 1) {
            name += "_step" + std::to_string(step);
        }
        const std::string reference = root + "/artifacts/" + name
            + "/reference";
        std::vector<float> expected_decode_mask;
        std::vector<float> expected_decode_cos;
        std::vector<float> expected_decode_sin;
        if (!load_binary(
                reference + "/layer0_attention_mask_f32.bin",
                present_length,
                expected_decode_mask)
            || !load_binary(
                reference + "/layer0_position_embeddings_0_f32.bin",
                4 * 128,
                expected_decode_cos)
            || !load_binary(
                reference + "/layer0_position_embeddings_1_f32.bin",
                4 * 128,
                expected_decode_sin)) {
            return EXIT_FAILURE;
        }
        decode_masks_exact = decode_masks_exact
            && decode_mask == expected_decode_mask;
        decode_cos_max = std::max(
            decode_cos_max,
            maximum_difference(decode_cos, expected_decode_cos));
        decode_sin_max = std::max(
            decode_sin_max,
            maximum_difference(decode_sin, expected_decode_sin));
    }

    std::cout << std::boolalpha
              << "input_ids_exact=" << ids_exact << '\n'
              << "attention_mask_exact=" << attention_exact << '\n'
              << "mm_token_type_ids_exact=" << mm_types_exact << '\n'
              << "position_ids_exact=" << positions_exact << '\n'
              << "causal_mask_exact=" << mask_exact << '\n'
              << "rope_cos_max_abs=" << cos_max << '\n'
              << "rope_sin_max_abs=" << sin_max << '\n'
              << "decode_masks_exact=" << decode_masks_exact << '\n'
              << "decode_rope_cos_max_abs=" << decode_cos_max << '\n'
              << "decode_rope_sin_max_abs=" << decode_sin_max << '\n'
              << "image_token_span=[" << actual.image_token_start << ", "
              << actual.image_token_end << ")\n";

    const bool passed = ids_exact && attention_exact && mm_types_exact
        && positions_exact && mask_exact && cos_max <= 2.0e-5
        && sin_max <= 2.0e-5 && decode_masks_exact
        && decode_cos_max <= 2.0e-5 && decode_sin_max <= 2.0e-5;
    std::cout << "prompt_mrope_contract_passed=" << passed << '\n';
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
