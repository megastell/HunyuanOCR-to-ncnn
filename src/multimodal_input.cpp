#include "detail/multimodal_input.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace {

constexpr int kPatchSize = 16;
constexpr int kMergeSize = 2;
constexpr int kPatchVectorSize = 768;
constexpr int kPatchCount = 1100;
constexpr int kVisionHiddenSize = 1152;
constexpr int kTextHiddenSize = 1024;
constexpr int kVisionBlocks = 27;
constexpr int kMergedTokens = 288;
constexpr int kPrefillLength = 313;
constexpr int kImageTokenId = 120120;
constexpr int kExpectedGridT = 1;
constexpr int kExpectedGridH = 22;
constexpr int kExpectedGridW = 50;
constexpr int kMinimumPixels = 262144;
constexpr int kMaximumPixels = 16777216;

constexpr float kImageMean[3] = {
    0.48145466f,
    0.4578275f,
    0.40821073f,
};
constexpr float kImageStd[3] = {
    0.26862954f,
    0.26130258f,
    0.27577711f,
};

constexpr std::size_t kPixelValueCount =
    static_cast<std::size_t>(kPatchCount) * kPatchVectorSize;
constexpr std::size_t kVisionHiddenCount =
    static_cast<std::size_t>(kPatchCount) * kVisionHiddenSize;
constexpr std::size_t kMergedHiddenCount =
    static_cast<std::size_t>(kMergedTokens) * kTextHiddenSize;
constexpr std::size_t kPrefillHiddenCount =
    static_cast<std::size_t>(kPrefillLength) * kTextHiddenSize;

std::size_t logical_count(const ncnn::Mat& value)
{
    return static_cast<std::size_t>(value.w)
        * static_cast<std::size_t>(value.h)
        * static_cast<std::size_t>(value.d)
        * static_cast<std::size_t>(value.c)
        * static_cast<std::size_t>(value.elempack);
}

bool unpack_mat(const ncnn::Mat& value, std::vector<float>& result)
{
    if (value.empty()
        || value.elemsize != sizeof(float)
            * static_cast<std::size_t>(value.elempack)) {
        return false;
    }
    result.assign(logical_count(value), 0.0f);
    const int pack = value.elempack;
    if (value.dims == 1) {
        const float* source = value;
        std::copy(source, source + result.size(), result.begin());
        return true;
    }
    if (value.dims == 2) {
        for (int y = 0; y < value.h; ++y) {
            const float* source = value.row(y);
            for (int x = 0; x < value.w; ++x) {
                for (int p = 0; p < pack; ++p) {
                    const std::size_t destination =
                        static_cast<std::size_t>(y * pack + p)
                            * static_cast<std::size_t>(value.w)
                        + static_cast<std::size_t>(x);
                    result[destination] = source[x * pack + p];
                }
            }
        }
        return true;
    }
    if (value.dims == 3) {
        for (int q = 0; q < value.c; ++q) {
            const float* source = value.channel(q);
            for (int y = 0; y < value.h; ++y) {
                for (int x = 0; x < value.w; ++x) {
                    for (int p = 0; p < pack; ++p) {
                        const std::size_t destination =
                            (static_cast<std::size_t>(q * pack + p)
                                    * static_cast<std::size_t>(value.h)
                                + static_cast<std::size_t>(y))
                                * static_cast<std::size_t>(value.w)
                            + static_cast<std::size_t>(x);
                        const std::size_t source_index =
                            (static_cast<std::size_t>(y)
                                    * static_cast<std::size_t>(value.w)
                                + static_cast<std::size_t>(x))
                                * static_cast<std::size_t>(pack)
                            + static_cast<std::size_t>(p);
                        result[destination] = source[source_index];
                    }
                }
            }
        }
        return true;
    }
    return false;
}

int round_to_multiple(double value, int multiple)
{
    return static_cast<int>(std::floor(value / multiple + 0.5)) * multiple;
}

bool smart_resize(
    int height,
    int width,
    int& resized_height,
    int& resized_width)
{
    if (height <= 0 || width <= 0
        || static_cast<double>(std::max(height, width))
                / std::min(height, width)
            > 200.0) {
        return false;
    }
    const int factor = kPatchSize * kMergeSize;
    resized_height = round_to_multiple(height, factor);
    resized_width = round_to_multiple(width, factor);
    const std::int64_t pixels = static_cast<std::int64_t>(resized_height)
        * resized_width;
    if (pixels > kMaximumPixels) {
        const double beta = std::sqrt(
            static_cast<double>(height) * width / kMaximumPixels);
        resized_height = std::max(
            factor,
            static_cast<int>(std::floor(height / beta / factor)) * factor);
        resized_width = std::max(
            factor,
            static_cast<int>(std::floor(width / beta / factor)) * factor);
    } else if (pixels < kMinimumPixels) {
        const double beta = std::sqrt(
            static_cast<double>(kMinimumPixels)
            / (static_cast<double>(height) * width));
        resized_height = static_cast<int>(
            std::ceil(height * beta / factor))
            * factor;
        resized_width = static_cast<int>(
            std::ceil(width * beta / factor))
            * factor;
    }
    return true;
}

double bicubic_kernel(double value)
{
    constexpr double a = -0.5;
    value = std::abs(value);
    if (value < 1.0) {
        return ((a + 2.0) * value - (a + 3.0))
            * value * value + 1.0;
    }
    if (value < 2.0) {
        return (((value - 5.0) * value + 8.0) * value - 4.0) * a;
    }
    return 0.0;
}

struct ResizeCoefficients {
    int kernel_size = 0;
    std::vector<int> starts;
    std::vector<int> counts;
    std::vector<std::int32_t> weights;
};

bool build_resize_coefficients(
    int input_size,
    int output_size,
    ResizeCoefficients& coefficients)
{
    constexpr double support = 2.0;
    constexpr int precision_bits = 22;
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double scaled_support = support * filter_scale;
    const double inverse_filter_scale = 1.0 / filter_scale;
    coefficients.kernel_size =
        static_cast<int>(std::ceil(scaled_support)) * 2 + 1;
    coefficients.starts.resize(output_size);
    coefficients.counts.resize(output_size);
    coefficients.weights.assign(
        static_cast<std::size_t>(output_size) * coefficients.kernel_size,
        0);

    std::vector<double> floating_weights(coefficients.kernel_size);
    for (int output = 0; output < output_size; ++output) {
        const double center = (output + 0.5) * scale;
        int minimum = static_cast<int>(center - scaled_support + 0.5);
        minimum = std::max(minimum, 0);
        int maximum = static_cast<int>(center + scaled_support + 0.5);
        maximum = std::min(maximum, input_size);
        const int count = maximum - minimum;
        if (count <= 0 || count > coefficients.kernel_size) {
            return false;
        }

        double weight_sum = 0.0;
        for (int index = 0; index < count; ++index) {
            const double weight = bicubic_kernel(
                (index + minimum - center + 0.5)
                * inverse_filter_scale);
            floating_weights[index] = weight;
            weight_sum += weight;
        }
        if (weight_sum == 0.0) {
            return false;
        }

        coefficients.starts[output] = minimum;
        coefficients.counts[output] = count;
        for (int index = 0; index < count; ++index) {
            const double normalized = floating_weights[index] / weight_sum;
            const double scaled = normalized * (1 << precision_bits);
            coefficients.weights[
                static_cast<std::size_t>(output)
                    * coefficients.kernel_size
                + index] = static_cast<std::int32_t>(
                    scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
        }
    }
    return true;
}

std::uint8_t clip_resized_channel(std::int64_t accumulator)
{
    constexpr int precision_bits = 22;
    const std::int64_t value = accumulator >> precision_bits;
    return static_cast<std::uint8_t>(std::clamp<std::int64_t>(value, 0, 255));
}

bool resize_rgb_pillow_bicubic(
    const std::uint8_t* source,
    int source_width,
    int source_height,
    int destination_width,
    int destination_height,
    std::vector<std::uint8_t>& destination)
{
    constexpr int channels = 3;
    constexpr int rounding = 1 << 21;
    ResizeCoefficients horizontal;
    ResizeCoefficients vertical;
    if (!build_resize_coefficients(
            source_width, destination_width, horizontal)
        || !build_resize_coefficients(
            source_height, destination_height, vertical)) {
        return false;
    }

    std::vector<std::uint8_t> intermediate(
        static_cast<std::size_t>(destination_width) * source_height * channels);
    for (int y = 0; y < source_height; ++y) {
        for (int x = 0; x < destination_width; ++x) {
            const int start = horizontal.starts[x];
            const int count = horizontal.counts[x];
            const std::int32_t* weights = horizontal.weights.data()
                + static_cast<std::size_t>(x) * horizontal.kernel_size;
            for (int channel = 0; channel < channels; ++channel) {
                std::int64_t accumulator = rounding;
                for (int index = 0; index < count; ++index) {
                    const std::size_t source_index =
                        (static_cast<std::size_t>(y) * source_width
                            + start + index)
                            * channels
                        + channel;
                    accumulator += source[source_index] * weights[index];
                }
                intermediate[
                    (static_cast<std::size_t>(y) * destination_width + x)
                        * channels
                    + channel] = clip_resized_channel(accumulator);
            }
        }
    }

    destination.resize(
        static_cast<std::size_t>(destination_width)
        * destination_height * channels);
    for (int y = 0; y < destination_height; ++y) {
        const int start = vertical.starts[y];
        const int count = vertical.counts[y];
        const std::int32_t* weights = vertical.weights.data()
            + static_cast<std::size_t>(y) * vertical.kernel_size;
        for (int x = 0; x < destination_width; ++x) {
            for (int channel = 0; channel < channels; ++channel) {
                std::int64_t accumulator = rounding;
                for (int index = 0; index < count; ++index) {
                    const std::size_t intermediate_index =
                        (static_cast<std::size_t>(start + index)
                                * destination_width
                            + x)
                            * channels
                        + channel;
                    accumulator +=
                        intermediate[intermediate_index] * weights[index];
                }
                destination[
                    (static_cast<std::size_t>(y) * destination_width + x)
                        * channels
                    + channel] = clip_resized_channel(accumulator);
            }
        }
    }
    return true;
}

bool preprocess_image(
    const std::string& image_path,
    std::vector<float>& pixel_values,
    std::vector<float>& original_rgb,
    std::vector<float>& resized_rgb,
    int& original_width,
    int& original_height,
    int& resized_width,
    int& resized_height)
{
    int channels = 0;
    stbi_uc* decoded = stbi_load(
        image_path.c_str(),
        &original_width,
        &original_height,
        &channels,
        3);
    if (decoded == nullptr) {
        std::cerr << "Unable to decode image: " << stbi_failure_reason() << '\n';
        return false;
    }
    if (!smart_resize(
            original_height,
            original_width,
            resized_height,
            resized_width)) {
        stbi_image_free(decoded);
        return false;
    }

    const std::size_t original_value_count =
        static_cast<std::size_t>(original_width) * original_height * 3;
    original_rgb.resize(original_value_count);
    for (std::size_t index = 0; index < original_value_count; ++index) {
        original_rgb[index] = static_cast<float>(decoded[index]);
    }

    std::vector<std::uint8_t> quantized;
    const bool resize_succeeded = resize_rgb_pillow_bicubic(
        decoded,
        original_width,
        original_height,
        resized_width,
        resized_height,
        quantized);
    stbi_image_free(decoded);
    if (!resize_succeeded) {
        return false;
    }

    const std::size_t resized_pixels =
        static_cast<std::size_t>(resized_width) * resized_height;
    resized_rgb.resize(resized_pixels * 3);
    for (std::size_t index = 0; index < quantized.size(); ++index) {
        resized_rgb[index] = static_cast<float>(quantized[index]);
    }

    const int grid_h = resized_height / kPatchSize;
    const int grid_w = resized_width / kPatchSize;
    if (grid_h != kExpectedGridH || grid_w != kExpectedGridW) {
        std::cerr << "Unexpected resized grid: " << grid_h << 'x' << grid_w
                  << '\n';
        return false;
    }

    pixel_values.resize(kPixelValueCount);
    for (int patch_y = 0; patch_y < grid_h; ++patch_y) {
        for (int patch_x = 0; patch_x < grid_w; ++patch_x) {
            const std::size_t patch =
                static_cast<std::size_t>(patch_y) * grid_w + patch_x;
            for (int channel = 0; channel < 3; ++channel) {
                for (int y = 0; y < kPatchSize; ++y) {
                    for (int x = 0; x < kPatchSize; ++x) {
                        const std::size_t source_index =
                            (static_cast<std::size_t>(
                                 patch_y * kPatchSize + y)
                                    * resized_width
                                + patch_x * kPatchSize + x)
                                * 3
                            + channel;
                        const float scaled =
                            static_cast<float>(quantized[source_index])
                            * (1.0f / 255.0f);
                        const std::size_t feature =
                            (static_cast<std::size_t>(channel) * kPatchSize
                                    + y)
                                * kPatchSize
                            + x;
                        pixel_values[patch * kPatchVectorSize + feature] =
                            (scaled - kImageMean[channel])
                            / kImageStd[channel];
                    }
                }
            }
        }
    }
    return true;
}

bool configure_and_load(
    ncnn::Net& network,
    const std::string& directory,
    const std::string& name,
    bool use_packing_layout,
    int num_threads)
{
    network.opt.use_vulkan_compute = false;
    network.opt.use_packing_layout = use_packing_layout;
    network.opt.num_threads = num_threads;
    return network.load_param(
            (directory + "/" + name + ".ncnn.param").c_str())
            == 0
        && network.load_model(
            (directory + "/" + name + ".ncnn.bin").c_str())
            == 0;
}

bool run_component(
    ncnn::Net& network,
    const ncnn::Mat& input,
    ncnn::Mat& output)
{
    ncnn::Extractor extractor = network.create_extractor();
    extractor.set_light_mode(false);
    if (extractor.input("in0", input) != 0) {
        return false;
    }
    ncnn::Mat raw_output;
    if (extractor.extract("out0", raw_output) != 0) {
        return false;
    }
    output = raw_output.clone();
    return !output.empty();
}

bool run_vision_tower(
    const std::string& model_directory,
    bool use_packing_layout,
    int num_threads,
    std::vector<float>& pixel_values,
    std::vector<float>& vision_embeddings)
{
    ncnn::Mat flat(
        static_cast<int>(pixel_values.size()), pixel_values.data());
    ncnn::Mat hidden = flat.reshape(kPatchVectorSize, kPatchCount).clone();
    if (hidden.empty()) {
        return false;
    }

    const std::string patch_name = "vision_patch_embed";
    ncnn::Net patch_network;
    if (!configure_and_load(
            patch_network,
            model_directory + "/" + patch_name,
            patch_name,
            use_packing_layout,
            num_threads)) {
        return false;
    }
    ncnn::Mat patch_output;
    if (!run_component(patch_network, hidden, patch_output)) {
        return false;
    }
    patch_network.clear();
    hidden = patch_output.reshape(kVisionHiddenSize, kPatchCount).clone();
    if (hidden.empty()) {
        return false;
    }

    for (int layer = 0; layer < kVisionBlocks; ++layer) {
        const std::string name = "vision_block" + std::to_string(layer);
        ncnn::Net network;
        if (!configure_and_load(
                network,
                model_directory + "/" + name,
                name,
                use_packing_layout,
                num_threads)) {
            return false;
        }
        ncnn::Mat output;
        if (!run_component(network, hidden, output)) {
            return false;
        }
        network.clear();
        hidden = output;
    }

    const std::string merger_name = "vision_patch_merger";
    ncnn::Net merger_network;
    if (!configure_and_load(
            merger_network,
            model_directory + "/" + merger_name,
            merger_name,
            use_packing_layout,
            num_threads)) {
        return false;
    }
    ncnn::Mat merger_input = hidden.reshape(
        kVisionHiddenSize, kPatchCount, 1).clone();
    ncnn::Mat merger_output;
    if (merger_input.empty()
        || !run_component(merger_network, merger_input, merger_output)) {
        return false;
    }
    merger_network.clear();
    return unpack_mat(merger_output, vision_embeddings)
        && vision_embeddings.size() == kMergedHiddenCount;
}

bool run_text_embedding(
    ncnn::Net& network,
    int token,
    std::vector<float>& embedding)
{
    ncnn::Mat token_input(1, static_cast<std::size_t>(4u));
    if (token_input.empty()) {
        return false;
    }
    int* token_pointer = token_input;
    token_pointer[0] = token;
    ncnn::Mat output;
    return run_component(network, token_input, output)
        && unpack_mat(output, embedding)
        && embedding.size() == kTextHiddenSize;
}

} // namespace

bool build_multimodal_prefill_input(
    const std::string& model_directory,
    const std::string& image_path,
    bool use_packing_layout,
    int num_threads,
    ncnn::Net& text_embedding_network,
    MultimodalPrefillInput& result)
{
    std::vector<float> pixel_values;
    std::vector<float> original_rgb;
    std::vector<float> resized_rgb;
    if (!preprocess_image(
            image_path,
            pixel_values,
            original_rgb,
            resized_rgb,
            result.original_width,
            result.original_height,
            result.resized_width,
            result.resized_height)) {
        std::cerr << "C++ image preprocessing failed\n";
        return false;
    }
    result.image_grid_thw = {
        kExpectedGridT,
        result.resized_height / kPatchSize,
        result.resized_width / kPatchSize};

    if (result.image_grid_thw[0] != kExpectedGridT
        || result.image_grid_thw[1] != kExpectedGridH
        || result.image_grid_thw[2] != kExpectedGridW) {
        std::cerr << "This runtime currently requires image grid [1,22,50]\n";
        return false;
    }

    std::vector<float> vision_embeddings;
    if (!run_vision_tower(
            model_directory,
            use_packing_layout,
            num_threads,
            pixel_values,
            vision_embeddings)) {
        std::cerr << "Full ncnn vision tower failed\n";
        return false;
    }
    if (!build_fixed_ocr_prompt_inputs(
            model_directory,
            result.image_grid_thw,
            result.prompt_inputs)) {
        return false;
    }

    std::vector<float> text_embeddings(kPrefillHiddenCount);
    result.image_token_start = result.prompt_inputs.image_token_start;
    result.image_token_end = result.prompt_inputs.image_token_end;
    int image_token_count = 0;
    for (int position = 0; position < kPrefillLength; ++position) {
        std::vector<float> embedding;
        if (!run_text_embedding(
                text_embedding_network,
                static_cast<int>(result.prompt_inputs.input_ids[position]),
                embedding)) {
            std::cerr << "Text embedding failed at position " << position
                      << '\n';
            return false;
        }
        std::copy(
            embedding.begin(),
            embedding.end(),
            text_embeddings.begin()
                + static_cast<std::size_t>(position) * kTextHiddenSize);
        if (result.prompt_inputs.input_ids[position] == kImageTokenId) {
            ++image_token_count;
        }
    }
    if (image_token_count != kMergedTokens
        || result.image_token_start != 2
        || result.image_token_end != 290) {
        std::cerr << "Unexpected image token span\n";
        return false;
    }
    result.hidden_states = text_embeddings;
    std::copy(
        vision_embeddings.begin(),
        vision_embeddings.end(),
        result.hidden_states.begin()
            + static_cast<std::size_t>(result.image_token_start)
                * kTextHiddenSize);
    return true;
}
