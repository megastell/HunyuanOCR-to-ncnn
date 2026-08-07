#include "detail/multimodal_input.h"
#include "detail/image_decode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kPatchSize = 16;
constexpr int kMergeSize = 2;
constexpr int kPatchVectorSize = 768;
constexpr int kVisionHiddenSize = 1152;
constexpr int kTextHiddenSize = 1024;
constexpr int kVisionBlocks = 27;
constexpr int kImageTokenId = 120120;
constexpr int kMinimumPixels = 262144;
constexpr int kMaximumPixels = 16777216;
constexpr int kPositionGridSize = 128;
constexpr int kMergerWidth = kVisionHiddenSize * kMergeSize * kMergeSize;

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
    int max_vision_patches,
    std::string& error,
    int& original_width,
    int& original_height,
    int& resized_width,
    int& resized_height)
{
    std::vector<std::uint8_t> decoded;
    if (!hunyuanocr::detail::decode_image_rgb(
            image_path,
            decoded,
            original_width,
            original_height,
            error)) {
        return false;
    }
    if (!smart_resize(
            original_height,
            original_width,
            resized_height,
            resized_width)) {
        error = "Image dimensions are invalid or exceed the 200:1 ratio limit";
        return false;
    }

    std::vector<std::uint8_t> quantized;
    const bool resize_succeeded = resize_rgb_pillow_bicubic(
        decoded.data(),
        original_width,
        original_height,
        resized_width,
        resized_height,
        quantized);
    decoded.clear();
    decoded.shrink_to_fit();
    if (!resize_succeeded) {
        error = "Pillow-compatible bicubic image resize failed";
        return false;
    }

    const int grid_h = resized_height / kPatchSize;
    const int grid_w = resized_width / kPatchSize;
    if (grid_h <= 0 || grid_w <= 0
        || grid_h % kMergeSize != 0 || grid_w % kMergeSize != 0) {
        error = "Unsupported resized image grid [1,"
            + std::to_string(grid_h) + ',' + std::to_string(grid_w)
            + "]; both spatial dimensions must be positive and even";
        return false;
    }

    const std::size_t patch_count =
        static_cast<std::size_t>(grid_h) * grid_w;
    if (patch_count > static_cast<std::size_t>(max_vision_patches)) {
        const double attention_mib =
            16.0 * patch_count * patch_count * sizeof(float)
            / (1024.0 * 1024.0);
        error = "Image grid [1," + std::to_string(grid_h) + ','
            + std::to_string(grid_w) + "] contains "
            + std::to_string(patch_count)
            + " vision patches, exceeding the configured limit of "
            + std::to_string(max_vision_patches)
            + ". Estimated FP32 attention-score memory is "
            + std::to_string(static_cast<long long>(attention_mib))
            + " MiB per vision layer. Resize the image or raise "
              "max_vision_patches only when sufficient RAM is available.";
        return false;
    }
    pixel_values.resize(patch_count * kPatchVectorSize);
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

bool load_f32_file(
    const std::string& path,
    std::size_t expected_count,
    std::vector<float>& values)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamsize bytes = file.tellg();
    if (bytes != static_cast<std::streamsize>(
            expected_count * sizeof(float))) {
        return false;
    }
    values.resize(expected_count);
    file.seekg(0, std::ios::beg);
    return static_cast<bool>(file.read(
        reinterpret_cast<char*>(values.data()), bytes));
}

bool add_interpolated_positions(
    const std::vector<float>& table,
    int grid_h,
    int grid_w,
    std::vector<float>& hidden)
{
    const std::size_t table_count =
        static_cast<std::size_t>(kPositionGridSize) * kPositionGridSize
        * kVisionHiddenSize;
    if (table.size() != table_count) {
        return false;
    }
    if (hidden.size()
        != static_cast<std::size_t>(grid_h) * grid_w
            * kVisionHiddenSize) {
        return false;
    }
    for (int y = 0; y < grid_h; ++y) {
        const float source_y =
            (static_cast<float>(y) + 0.5f) * kPositionGridSize / grid_h
            - 0.5f;
        const int floor_y = static_cast<int>(std::floor(source_y));
        const int y0 = std::clamp(floor_y, 0, kPositionGridSize - 1);
        const int y1 = std::clamp(floor_y + 1, 0, kPositionGridSize - 1);
        const float weight_y = source_y - floor_y;
        for (int x = 0; x < grid_w; ++x) {
            const float source_x =
                (static_cast<float>(x) + 0.5f) * kPositionGridSize / grid_w
                - 0.5f;
            const int floor_x = static_cast<int>(std::floor(source_x));
            const int x0 = std::clamp(floor_x, 0, kPositionGridSize - 1);
            const int x1 = std::clamp(floor_x + 1, 0, kPositionGridSize - 1);
            const float weight_x = source_x - floor_x;
            const std::size_t destination =
                (static_cast<std::size_t>(y) * grid_w + x)
                * kVisionHiddenSize;
            const std::size_t source00 =
                (static_cast<std::size_t>(y0) * kPositionGridSize + x0)
                * kVisionHiddenSize;
            const std::size_t source01 =
                (static_cast<std::size_t>(y0) * kPositionGridSize + x1)
                * kVisionHiddenSize;
            const std::size_t source10 =
                (static_cast<std::size_t>(y1) * kPositionGridSize + x0)
                * kVisionHiddenSize;
            const std::size_t source11 =
                (static_cast<std::size_t>(y1) * kPositionGridSize + x1)
                * kVisionHiddenSize;
            for (int feature = 0; feature < kVisionHiddenSize; ++feature) {
                const float top = table[source00 + feature]
                    + (table[source01 + feature] - table[source00 + feature])
                        * weight_x;
                const float bottom = table[source10 + feature]
                    + (table[source11 + feature] - table[source10 + feature])
                        * weight_x;
                hidden[destination + feature] +=
                    top + (bottom - top) * weight_y;
            }
        }
    }
    return true;
}

bool run_named_component(
    const std::string& model_directory,
    const std::string& name,
    bool use_packing_layout,
    int num_threads,
    const ncnn::Mat& input,
    ncnn::Mat& output)
{
    ncnn::Net network;
    if (!configure_and_load(
            network,
            model_directory + "/" + name,
            name,
            use_packing_layout,
            num_threads)
        || !run_component(network, input, output)) {
        return false;
    }
    network.clear();
    return true;
}

bool run_vision_tower(
    const std::string& model_directory,
    bool use_packing_layout,
    int num_threads,
    const std::vector<float>& pixel_values,
    int grid_h,
    int grid_w,
    const MultimodalResources& resources,
    std::vector<float>& vision_embeddings)
{
    const int patch_count = grid_h * grid_w;
    if (patch_count <= 0
        || pixel_values.size()
            != static_cast<std::size_t>(patch_count) * kPatchVectorSize) {
        return false;
    }
    ncnn::Mat flat(
        static_cast<int>(pixel_values.size()),
        const_cast<float*>(pixel_values.data()));
    ncnn::Mat input = flat.reshape(kPatchVectorSize, patch_count).clone();
    ncnn::Mat output;
    if (input.empty()
        || !run_named_component(
            model_directory,
            "vision_patch_embed",
            use_packing_layout,
            num_threads,
            input,
            output)) {
        return false;
    }
    std::vector<float> hidden_values;
    if (!unpack_mat(output, hidden_values)
        || !add_interpolated_positions(
            resources.vision_position_embedding,
            grid_h,
            grid_w,
            hidden_values)) {
        return false;
    }
    ncnn::Mat hidden(
        kVisionHiddenSize, patch_count, hidden_values.data());
    hidden = hidden.clone();
    input.release();
    hidden_values.clear();
    hidden_values.shrink_to_fit();

    for (int layer = 0; layer < kVisionBlocks; ++layer) {
        const std::string name = "vision_block" + std::to_string(layer);
        if (!run_named_component(
                model_directory,
                name,
                use_packing_layout,
                num_threads,
                hidden,
                output)) {
            return false;
        }
        hidden = output;
    }

    if (!run_named_component(
            model_directory,
            "vision_patch_merger_pre_rms",
            use_packing_layout,
            num_threads,
            hidden,
            output)
        || !unpack_mat(output, hidden_values)) {
        return false;
    }
    hidden.release();
    output.release();
    ncnn::Mat convolution_input(grid_w, grid_h, kVisionHiddenSize);
    if (convolution_input.empty()) return false;
    for (int feature = 0; feature < kVisionHiddenSize; ++feature) {
        float* channel = convolution_input.channel(feature);
        for (int patch = 0; patch < patch_count; ++patch) {
            channel[patch] = hidden_values[
                static_cast<std::size_t>(patch) * kVisionHiddenSize + feature];
        }
    }
    hidden_values.clear();
    hidden_values.shrink_to_fit();
    if (!run_named_component(
            model_directory,
            "vision_patch_merger_conv",
            use_packing_layout,
            num_threads,
            convolution_input,
            output)) {
        return false;
    }
    std::vector<float> convolution_values;
    if (!unpack_mat(output, convolution_values)) return false;
    convolution_input.release();
    output.release();

    constexpr std::size_t newline_count = kMergerWidth;
    constexpr std::size_t boundary_count = kTextHiddenSize;
    const std::vector<float>& constants = resources.vision_merger_constants;
    if (constants.size() != newline_count + 2 * boundary_count) {
        return false;
    }
    const int merged_h = grid_h / kMergeSize;
    const int merged_w = grid_w / kMergeSize;
    const int projection_tokens = merged_h * (merged_w + 1);
    std::vector<float> projection_values(
        static_cast<std::size_t>(projection_tokens) * kMergerWidth);
    for (int y = 0; y < merged_h; ++y) {
        for (int x = 0; x <= merged_w; ++x) {
            const std::size_t destination =
                static_cast<std::size_t>(y * (merged_w + 1) + x)
                * kMergerWidth;
            for (int feature = 0; feature < kMergerWidth; ++feature) {
                projection_values[destination + feature] = x == merged_w
                    ? constants[feature]
                    : convolution_values[
                        (static_cast<std::size_t>(feature) * merged_h + y)
                            * merged_w + x];
            }
        }
    }
    ncnn::Mat projection_input(
        kMergerWidth, projection_tokens, projection_values.data());
    if (!run_named_component(
            model_directory,
            "vision_patch_merger_projection",
            use_packing_layout,
            num_threads,
            projection_input,
            output)) {
        return false;
    }
    projection_input.release();
    projection_values.clear();
    projection_values.shrink_to_fit();
    convolution_values.clear();
    convolution_values.shrink_to_fit();
    std::vector<float> projected;
    if (!unpack_mat(output, projected)) return false;
    output.release();
    const int image_tokens = projection_tokens + 2;
    std::vector<float> with_boundaries(
        static_cast<std::size_t>(image_tokens) * kTextHiddenSize);
    std::copy(
        constants.begin() + newline_count,
        constants.begin() + newline_count + boundary_count,
        with_boundaries.begin());
    std::copy(
        projected.begin(),
        projected.end(),
        with_boundaries.begin() + kTextHiddenSize);
    std::copy(
        constants.begin() + newline_count + boundary_count,
        constants.end(),
        with_boundaries.end() - kTextHiddenSize);
    projected.clear();
    projected.shrink_to_fit();
    ncnn::Mat post_input(
        kTextHiddenSize, image_tokens, with_boundaries.data());
    if (!run_named_component(
            model_directory,
            "vision_patch_merger_post_rms",
            use_packing_layout,
            num_threads,
            post_input,
            output)
        || !unpack_mat(output, vision_embeddings)) {
        return false;
    }
    return vision_embeddings.size()
        == static_cast<std::size_t>(image_tokens) * kTextHiddenSize;
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

bool load_multimodal_resources(
    const std::string& model_directory,
    MultimodalResources& resources,
    std::string& error)
{
    const std::size_t position_count =
        static_cast<std::size_t>(kPositionGridSize) * kPositionGridSize
        * kVisionHiddenSize;
    constexpr std::size_t merger_count =
        kMergerWidth + 2 * kTextHiddenSize;
    resources = MultimodalResources{};
    if (!load_f32_file(
            model_directory
                + "/vision_patch_embed/vision_position_embedding.f32.bin",
            position_count,
            resources.vision_position_embedding)) {
        error = "Unable to load the 128x128 vision position table";
        return false;
    }
    if (!load_f32_file(
            model_directory
                + "/vision_patch_merger/vision_patch_merger_constants.f32.bin",
            merger_count,
            resources.vision_merger_constants)) {
        error = "Unable to load the vision merger constants";
        return false;
    }
    if (!load_ocr_prompt_token_ids(
            model_directory, resources.prompt_token_ids)) {
        error = "Unable to construct the fixed OCR prompt token IDs";
        return false;
    }
    return true;
}

bool build_multimodal_prefill_input(
    const std::string& model_directory,
    const std::string& image_path,
    bool use_packing_layout,
    int num_threads,
    int max_vision_patches,
    ncnn::Net& text_embedding_network,
    const MultimodalResources& resources,
    std::string& error,
    MultimodalPrefillInput& result)
{
    std::vector<float> pixel_values;
    if (!preprocess_image(
            image_path,
            pixel_values,
            max_vision_patches,
            error,
            result.original_width,
            result.original_height,
            result.resized_width,
            result.resized_height)) {
        return false;
    }
    result.image_grid_thw = {
        1,
        result.resized_height / kPatchSize,
        result.resized_width / kPatchSize};
    const int grid_h = static_cast<int>(result.image_grid_thw[1]);
    const int grid_w = static_cast<int>(result.image_grid_thw[2]);

    std::vector<float> vision_embeddings;
    if (!run_vision_tower(
            model_directory,
            use_packing_layout,
            num_threads,
            pixel_values,
            grid_h,
            grid_w,
            resources,
            vision_embeddings)) {
        error = "Full ncnn vision tower failed";
        return false;
    }
    if (!build_ocr_prompt_inputs(
            resources.prompt_token_ids,
            result.image_grid_thw,
            result.prompt_inputs)) {
        error = "Dynamic OCR prompt or mRoPE construction failed";
        return false;
    }

    const int prefill_length = static_cast<int>(
        result.prompt_inputs.input_ids.size());
    std::vector<float> text_embeddings(
        static_cast<std::size_t>(prefill_length) * kTextHiddenSize);
    result.image_token_start = result.prompt_inputs.image_token_start;
    result.image_token_end = result.prompt_inputs.image_token_end;
    int image_token_count = 0;
    for (int position = 0; position < prefill_length; ++position) {
        std::vector<float> embedding;
        if (!run_text_embedding(
                text_embedding_network,
                static_cast<int>(result.prompt_inputs.input_ids[position]),
                embedding)) {
            error = "Text embedding failed at position "
                + std::to_string(position);
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
    const int expected_image_tokens =
        (grid_h / kMergeSize) * (grid_w / kMergeSize + 1) + 2;
    if (image_token_count != expected_image_tokens
        || result.image_token_start != 2
        || result.image_token_end
            != result.image_token_start + expected_image_tokens
        || vision_embeddings.size()
            != static_cast<std::size_t>(expected_image_tokens)
                * kTextHiddenSize) {
        error = "Vision output and dynamic image-token span disagree";
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
