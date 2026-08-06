#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <net.h>

namespace {

constexpr int kRuntimeThreads = 9;
constexpr int kVisionBlocks = 27;
constexpr int kPatchCount = 1100;
constexpr int kPatchVectorSize = 768;
constexpr int kVisionHiddenSize = 1152;
constexpr int kTextHiddenSize = 1024;
constexpr int kGridT = 1;
constexpr int kGridH = 22;
constexpr int kGridW = 50;
constexpr int kMergedTokenCount = 288;

constexpr std::size_t kPixelCount =
    static_cast<std::size_t>(kPatchCount) * kPatchVectorSize;
constexpr std::size_t kVisionHiddenCount =
    static_cast<std::size_t>(kPatchCount) * kVisionHiddenSize;
constexpr std::size_t kMergedHiddenCount =
    static_cast<std::size_t>(kMergedTokenCount) * kTextHiddenSize;

struct Metrics {
    double maximum_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double cosine_similarity = 0.0;
    double maximum_expected_abs = 0.0;
    double actual_at_maximum_error = 0.0;
    double expected_at_maximum_error = 0.0;
    std::size_t maximum_error_index = 0;
};

template <typename T>
bool load_exact_binary(
    const std::string& path,
    std::size_t expected_count,
    std::vector<T>& values)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Unable to open " << path << '\n';
        return false;
    }
    const std::streamsize expected_bytes =
        static_cast<std::streamsize>(expected_count * sizeof(T));
    if (file.tellg() != expected_bytes) {
        std::cerr << "Unexpected file size for " << path
                  << ": actual=" << file.tellg()
                  << ", expected=" << expected_bytes << '\n';
        return false;
    }
    file.seekg(0, std::ios::beg);
    values.resize(expected_count);
    return static_cast<bool>(file.read(
        reinterpret_cast<char*>(values.data()), expected_bytes));
}

bool save_exact_binary(
    const std::string& path,
    const std::vector<float>& values)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Unable to write " << path << '\n';
        return false;
    }
    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return static_cast<bool>(file);
}

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
    if (value.dims == 4) {
        for (int q = 0; q < value.c; ++q) {
            const ncnn::Mat channel = value.channel(q);
            for (int z = 0; z < value.d; ++z) {
                const float* source = channel.depth(z);
                for (int y = 0; y < value.h; ++y) {
                    for (int x = 0; x < value.w; ++x) {
                        for (int p = 0; p < pack; ++p) {
                            const std::size_t destination =
                                (((static_cast<std::size_t>(q * pack + p)
                                    * static_cast<std::size_t>(value.d)
                                    + static_cast<std::size_t>(z))
                                    * static_cast<std::size_t>(value.h)
                                    + static_cast<std::size_t>(y))
                                    * static_cast<std::size_t>(value.w)
                                    + static_cast<std::size_t>(x));
                            const std::size_t source_index =
                                ((static_cast<std::size_t>(y)
                                    * static_cast<std::size_t>(value.w)
                                    + static_cast<std::size_t>(x))
                                    * static_cast<std::size_t>(pack)
                                    + static_cast<std::size_t>(p));
                            result[destination] = source[source_index];
                        }
                    }
                }
            }
        }
        return true;
    }
    return false;
}

bool calculate_metrics(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    Metrics& metrics)
{
    if (actual.size() != expected.size() || actual.empty()) {
        return false;
    }
    long double absolute_sum = 0.0;
    long double dot = 0.0;
    long double actual_norm = 0.0;
    long double expected_norm = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double a = actual[index];
        const double e = expected[index];
        if (!std::isfinite(a) || !std::isfinite(e)) {
            return false;
        }
        const double difference = std::abs(a - e);
        metrics.maximum_expected_abs = std::max(
            metrics.maximum_expected_abs, std::abs(e));
        if (difference > metrics.maximum_abs_error) {
            metrics.maximum_abs_error = difference;
            metrics.actual_at_maximum_error = a;
            metrics.expected_at_maximum_error = e;
            metrics.maximum_error_index = index;
        }
        absolute_sum += difference;
        dot += static_cast<long double>(a) * e;
        actual_norm += static_cast<long double>(a) * a;
        expected_norm += static_cast<long double>(e) * e;
    }
    metrics.mean_abs_error = static_cast<double>(
        absolute_sum / static_cast<long double>(actual.size()));
    const long double denominator = std::sqrt(actual_norm * expected_norm);
    metrics.cosine_similarity = denominator == 0.0
        ? (actual == expected ? 1.0 : 0.0)
        : static_cast<double>(dot / denominator);
    metrics.cosine_similarity = std::clamp(
        metrics.cosine_similarity, -1.0, 1.0);
    return true;
}

bool compare_mat(
    const ncnn::Mat& actual_mat,
    const std::vector<float>& expected,
    Metrics& metrics)
{
    if (logical_count(actual_mat) != expected.size()) {
        std::cerr << "Unexpected ncnn output shape: dims=" << actual_mat.dims
                  << ", w=" << actual_mat.w
                  << ", h=" << actual_mat.h
                  << ", d=" << actual_mat.d
                  << ", c=" << actual_mat.c
                  << ", elempack=" << actual_mat.elempack
                  << ", actual_count=" << logical_count(actual_mat)
                  << ", expected_count=" << expected.size() << '\n';
        return false;
    }
    std::vector<float> actual;
    if (!unpack_mat(actual_mat, actual)) {
        std::cerr << "Unable to unpack ncnn output: dims=" << actual_mat.dims
                  << ", w=" << actual_mat.w
                  << ", h=" << actual_mat.h
                  << ", d=" << actual_mat.d
                  << ", c=" << actual_mat.c
                  << ", elempack=" << actual_mat.elempack
                  << ", elemsize=" << actual_mat.elemsize
                  << ", logical_count=" << logical_count(actual_mat)
                  << '\n';
        return false;
    }
    if (!calculate_metrics(actual, expected, metrics)) {
        std::cerr << "Unable to compare ncnn output: actual_count="
                  << actual.size() << ", expected_count=" << expected.size()
                  << '\n';
        return false;
    }
    return true;
}

bool vision_hidden_passed(const Metrics& metrics)
{
    const double scaled_maximum = std::max(
        1.0, metrics.maximum_expected_abs * 5.0e-2);
    return metrics.maximum_abs_error <= scaled_maximum
        && metrics.mean_abs_error <= 2.0e-2
        && metrics.cosine_similarity >= 0.99999;
}

bool merged_hidden_passed(const Metrics& metrics)
{
    return metrics.maximum_abs_error <= 1.0e-1
        && metrics.mean_abs_error <= 1.0e-3
        && metrics.cosine_similarity >= 0.99999;
}

ncnn::Mat make_matrix(
    std::vector<float>& values,
    int width,
    int height)
{
    ncnn::Mat flat(static_cast<int>(values.size()), values.data());
    return flat.reshape(width, height).clone();
}

bool configure_and_load(
    ncnn::Net& network,
    const std::string& directory,
    const std::string& name,
    bool use_packing_layout)
{
    network.opt.use_vulkan_compute = false;
    network.opt.use_packing_layout = use_packing_layout;
    network.opt.num_threads = kRuntimeThreads;
    return network.load_param(
            (directory + "/" + name + ".ncnn.param").c_str()) == 0
        && network.load_model(
            (directory + "/" + name + ".ncnn.bin").c_str()) == 0;
}

bool run_component(
    ncnn::Net& network,
    const ncnn::Mat& input,
    ncnn::Mat& output)
{
    ncnn::Extractor extractor = network.create_extractor();
    extractor.set_light_mode(false);
    const int input_status = extractor.input("in0", input);
    if (input_status != 0) {
        std::cerr << "ncnn input failed: status=" << input_status
                  << ", dims=" << input.dims
                  << ", w=" << input.w
                  << ", h=" << input.h
                  << ", d=" << input.d
                  << ", c=" << input.c
                  << ", elempack=" << input.elempack
                  << ", elemsize=" << input.elemsize << '\n';
        return false;
    }
    ncnn::Mat raw_output;
    const int extract_status = extractor.extract("out0", raw_output);
    if (extract_status != 0) {
        std::cerr << "ncnn extract failed: status=" << extract_status
                  << ", input_dims=" << input.dims
                  << ", input_w=" << input.w
                  << ", input_h=" << input.h
                  << ", input_d=" << input.d
                  << ", input_c=" << input.c
                  << ", input_elempack=" << input.elempack
                  << ", input_elemsize=" << input.elemsize << '\n';
        return false;
    }
    output = raw_output.clone();
    return !output.empty();
}

std::string reference_directory(
    const std::string& project_root,
    const std::string& name)
{
    return project_root + "/artifacts/" + name + "/reference";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <project_root> <packing:0|1>\n";
        return EXIT_FAILURE;
    }
    const std::string project_root = argv[1];
    const std::string packing_text = argv[2];
    if (packing_text != "0" && packing_text != "1") {
        return EXIT_FAILURE;
    }
    const bool use_packing_layout = packing_text == "1";

    std::cout << "===== HunyuanOCR full ncnn vision tower =====\n"
              << "packing layout   : " << std::boolalpha
              << use_packing_layout << '\n'
              << "pixel input      : [1100, 768]\n"
              << "image_grid_thw   : [1, 22, 50]\n"
              << "vision blocks    : 27\n"
              << "vision RoPE      : none\n"
              << "merged output    : [1, 288, 1024]\n\n";

    const std::string patch_name = "vision_patch_embed";
    const std::string patch_reference =
        reference_directory(project_root, patch_name);
    std::vector<float> pixel_values;
    std::vector<std::int64_t> image_grid_thw;
    std::vector<float> expected_patch_output;
    if (!load_exact_binary(
            patch_reference + "/pixel_values_f32.bin",
            kPixelCount,
            pixel_values)
        || !load_exact_binary(
            patch_reference + "/image_grid_thw_i64.bin",
            3,
            image_grid_thw)
        || !load_exact_binary(
            patch_reference + "/expected_output_f32.bin",
            kVisionHiddenCount,
            expected_patch_output)) {
        return EXIT_FAILURE;
    }
    if (image_grid_thw != std::vector<std::int64_t>{kGridT, kGridH, kGridW}
        || image_grid_thw[0] * image_grid_thw[1] * image_grid_thw[2]
            != kPatchCount) {
        std::cerr << "Unsupported image_grid_thw contract\n";
        return EXIT_FAILURE;
    }

    ncnn::Mat hidden = make_matrix(
        pixel_values, kPatchVectorSize, kPatchCount);
    ncnn::Net patch_network;
    const std::string patch_directory =
        project_root + "/artifacts/" + patch_name;
    if (!configure_and_load(
            patch_network,
            patch_directory,
            patch_name,
            use_packing_layout)) {
        std::cerr << "Failed to load patch embedding\n";
        return EXIT_FAILURE;
    }
    ncnn::Mat patch_output;
    if (!run_component(patch_network, hidden, patch_output)) {
        std::cerr << "Patch embedding inference failed\n";
        return EXIT_FAILURE;
    }
    patch_network.clear();
    Metrics patch_metrics;
    if (!compare_mat(patch_output, expected_patch_output, patch_metrics)
        || !vision_hidden_passed(patch_metrics)) {
        std::cerr << "Patch embedding parity failed: max="
                  << patch_metrics.maximum_abs_error << '\n';
        return EXIT_FAILURE;
    }
    std::cout << std::scientific << std::setprecision(3)
              << "Patch embedding: max=" << patch_metrics.maximum_abs_error
              << ", mean=" << patch_metrics.mean_abs_error
              << ", cos=" << patch_metrics.cosine_similarity << '\n';
    hidden = patch_output.reshape(kVisionHiddenSize, kPatchCount).clone();
    if (hidden.empty()) {
        std::cerr << "Unable to normalize patch output to [1100, 1152]\n";
        return EXIT_FAILURE;
    }

    double maximum_input_error = 0.0;
    double maximum_output_error = patch_metrics.maximum_abs_error;
    for (int layer = 0; layer < kVisionBlocks; ++layer) {
        const std::string name = "vision_block" + std::to_string(layer);
        const std::string reference = reference_directory(project_root, name);
        std::vector<float> expected_input;
        std::vector<float> expected_output;
        if (!load_exact_binary(
                reference + "/hidden_states_f32.bin",
                kVisionHiddenCount,
                expected_input)
            || !load_exact_binary(
                reference + "/expected_output_f32.bin",
                kVisionHiddenCount,
                expected_output)) {
            return EXIT_FAILURE;
        }

        Metrics input_metrics;
        if (!compare_mat(hidden, expected_input, input_metrics)
            || !vision_hidden_passed(input_metrics)) {
            std::cerr << "Vision block " << layer
                      << " input parity failed: max="
                      << input_metrics.maximum_abs_error << '\n';
            return EXIT_FAILURE;
        }

        ncnn::Net block_network;
        const std::string directory = project_root + "/artifacts/" + name;
        if (!configure_and_load(
                block_network,
                directory,
                name,
                use_packing_layout)) {
            std::cerr << "Failed to load vision block " << layer << '\n';
            return EXIT_FAILURE;
        }
        ncnn::Mat next_hidden;
        if (!run_component(block_network, hidden, next_hidden)) {
            std::cerr << "Vision block " << layer << " inference failed\n";
            return EXIT_FAILURE;
        }
        block_network.clear();

        Metrics output_metrics;
        if (!compare_mat(next_hidden, expected_output, output_metrics)
            || !vision_hidden_passed(output_metrics)) {
            std::cerr << "Vision block " << layer
                      << " output parity failed: max="
                      << output_metrics.maximum_abs_error
                      << ", mean=" << output_metrics.mean_abs_error
                      << ", cos=" << output_metrics.cosine_similarity
                      << ", expected_abs_max="
                      << output_metrics.maximum_expected_abs
                      << ", max_error_index="
                      << output_metrics.maximum_error_index
                      << ", actual_at_max="
                      << output_metrics.actual_at_maximum_error
                      << ", expected_at_max="
                      << output_metrics.expected_at_maximum_error << '\n';
            return EXIT_FAILURE;
        }
        maximum_input_error = std::max(
            maximum_input_error, input_metrics.maximum_abs_error);
        maximum_output_error = std::max(
            maximum_output_error, output_metrics.maximum_abs_error);
        std::cout << "Vision Block " << std::setw(2) << layer
                  << ": input_max=" << input_metrics.maximum_abs_error
                  << ", output_max=" << output_metrics.maximum_abs_error
                  << ", output_mean=" << output_metrics.mean_abs_error
                  << ", cos=" << output_metrics.cosine_similarity << '\n';
        hidden = next_hidden;
    }

    const std::string merger_name = "vision_patch_merger";
    const std::string merger_reference =
        reference_directory(project_root, merger_name);
    std::vector<float> expected_merger_input;
    std::vector<float> expected_merger_output;
    if (!load_exact_binary(
            merger_reference + "/hidden_states_f32.bin",
            kVisionHiddenCount,
            expected_merger_input)
        || !load_exact_binary(
            merger_reference + "/expected_output_f32.bin",
            kMergedHiddenCount,
            expected_merger_output)) {
        return EXIT_FAILURE;
    }
    Metrics merger_input_metrics;
    if (!compare_mat(hidden, expected_merger_input, merger_input_metrics)
        || !vision_hidden_passed(merger_input_metrics)) {
        std::cerr << "Patch merger input parity failed\n";
        return EXIT_FAILURE;
    }

    ncnn::Net merger_network;
    const std::string merger_directory =
        project_root + "/artifacts/" + merger_name;
    if (!configure_and_load(
            merger_network,
            merger_directory,
            merger_name,
            use_packing_layout)) {
        std::cerr << "Failed to load patch merger\n";
        return EXIT_FAILURE;
    }
    ncnn::Mat merger_hidden = hidden.reshape(
        kVisionHiddenSize, kPatchCount, 1).clone();
    if (merger_hidden.empty()) {
        std::cerr << "Unable to restore merger input to [1, 1100, 1152]\n";
        return EXIT_FAILURE;
    }
    ncnn::Mat merged_output;
    if (!run_component(merger_network, merger_hidden, merged_output)) {
        std::cerr << "Patch merger inference failed\n";
        return EXIT_FAILURE;
    }
    merger_network.clear();

    Metrics merger_metrics;
    std::vector<float> actual_merged_output;
    if (!compare_mat(merged_output, expected_merger_output, merger_metrics)
        || !merged_hidden_passed(merger_metrics)
        || !unpack_mat(merged_output, actual_merged_output)) {
        std::cerr << "Patch merger output parity failed: max="
                  << merger_metrics.maximum_abs_error
                  << ", mean=" << merger_metrics.mean_abs_error
                  << ", cos=" << merger_metrics.cosine_similarity << '\n';
        return EXIT_FAILURE;
    }

    const std::string output_directory =
        project_root + "/artifacts/vision_tower_full/output";
    std::filesystem::create_directories(output_directory);
    const std::string output_path = output_directory
        + "/vision_embeddings_"
        + (use_packing_layout ? "packed" : "unpacked")
        + "_f32.bin";
    if (!save_exact_binary(output_path, actual_merged_output)) {
        return EXIT_FAILURE;
    }

    std::cout << "\n===== Full vision result =====\n"
              << "Maximum block input error : " << maximum_input_error << '\n'
              << "Maximum block output error: " << maximum_output_error << '\n'
              << "Merger input error        : "
              << merger_input_metrics.maximum_abs_error << '\n'
              << "Merger output error       : "
              << merger_metrics.maximum_abs_error << '\n'
              << "Merger output mean error  : "
              << merger_metrics.mean_abs_error << '\n'
              << "Merger output cosine      : "
              << merger_metrics.cosine_similarity << '\n'
              << "Vision embedding shape    : [1, 288, 1024]\n"
              << "Vision embedding path     : " << output_path << "\n\n"
              << "Full ncnn vision tower passed.\n";
    return EXIT_SUCCESS;
}
