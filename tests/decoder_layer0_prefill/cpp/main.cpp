#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <net.h>

namespace {

constexpr int kSequenceLength = 313;
constexpr int kHiddenSize = 1024;
constexpr int kHeadDim = 128;
constexpr int kRopeComponents = 4;

constexpr std::size_t kHiddenCount =
    static_cast<std::size_t>(kSequenceLength) *
    static_cast<std::size_t>(kHiddenSize);

constexpr std::size_t kMaskCount =
    static_cast<std::size_t>(kSequenceLength) *
    static_cast<std::size_t>(kSequenceLength);

constexpr std::size_t kRopeCount =
    static_cast<std::size_t>(kRopeComponents) *
    static_cast<std::size_t>(kSequenceLength) *
    static_cast<std::size_t>(kHeadDim);

template <typename T>
bool load_exact_binary(
    const std::string& path,
    const std::size_t expected_count,
    std::vector<T>& values)
{
    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!file.is_open()) {
        std::cerr
            << "无法打开文件："
            << path
            << '\n';

        return false;
    }

    const std::streamsize actual_bytes =
        file.tellg();

    const std::streamsize expected_bytes =
        static_cast<std::streamsize>(
            expected_count * sizeof(T)
        );

    if (actual_bytes != expected_bytes) {
        std::cerr
            << "文件大小不符合预期："
            << path
            << "\n实际字节数："
            << actual_bytes
            << "\n预期字节数："
            << expected_bytes
            << '\n';

        return false;
    }

    file.seekg(0, std::ios::beg);

    values.resize(expected_count);

    if (!file.read(
            reinterpret_cast<char*>(values.data()),
            expected_bytes
        )) {
        std::cerr
            << "读取文件失败："
            << path
            << '\n';

        return false;
    }

    return true;
}

std::size_t logical_count(
    const ncnn::Mat& value)
{
    return
        static_cast<std::size_t>(value.w) *
        static_cast<std::size_t>(value.h) *
        static_cast<std::size_t>(value.d) *
        static_cast<std::size_t>(value.c) *
        static_cast<std::size_t>(value.elempack);
}

void print_shape(
    const std::string& name,
    const ncnn::Mat& value)
{
    std::cout
        << name
        << ": dims=" << value.dims
        << ", w=" << value.w
        << ", h=" << value.h
        << ", d=" << value.d
        << ", c=" << value.c
        << ", elempack=" << value.elempack
        << ", logical_count="
        << logical_count(value)
        << '\n';
}

bool is_valid_mat(
    const std::string& name,
    const ncnn::Mat& value,
    const std::size_t expected_count)
{
    if (value.empty()) {
        std::cerr
            << name
            << " 创建失败或为空。\n";

        return false;
    }

    const std::size_t count =
        logical_count(value);

    if (count != expected_count) {
        std::cerr
            << name
            << " 元素数量错误，实际="
            << count
            << "，预期="
            << expected_count
            << '\n';

        return false;
    }

    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 8) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <param>"
            << " <bin>"
            << " <hidden_states_f32.bin>"
            << " <attention_mask_f32.bin>"
            << " <rope_cos_f32.bin>"
            << " <rope_sin_f32.bin>"
            << " <expected_output_f32.bin>\n";

        return EXIT_FAILURE;
    }

    const std::string param_path = argv[1];
    const std::string model_path = argv[2];
    const std::string hidden_path = argv[3];
    const std::string mask_path = argv[4];
    const std::string rope_cos_path = argv[5];
    const std::string rope_sin_path = argv[6];
    const std::string expected_path = argv[7];

    std::vector<float> hidden_values;
    std::vector<float> mask_values;
    std::vector<float> rope_cos_values;
    std::vector<float> rope_sin_values;
    std::vector<float> expected_values;

    if (!load_exact_binary(
            hidden_path,
            kHiddenCount,
            hidden_values
        ) ||
        !load_exact_binary(
            mask_path,
            kMaskCount,
            mask_values
        ) ||
        !load_exact_binary(
            rope_cos_path,
            kRopeCount,
            rope_cos_values
        ) ||
        !load_exact_binary(
            rope_sin_path,
            kRopeCount,
            rope_sin_values
        ) ||
        !load_exact_binary(
            expected_path,
            kHiddenCount,
            expected_values
        )) {
        return EXIT_FAILURE;
    }

    /*
     * 直接引用vector持有的连续内存，再reshape成pnnx期望的形状。
     * vector必须在Extractor完成前保持存活。
     */
    ncnn::Mat hidden_flat(
        static_cast<int>(kHiddenCount),
        static_cast<void*>(hidden_values.data())
    );

    ncnn::Mat hidden_states =
        hidden_flat.reshape(
            kHiddenSize,
            kSequenceLength
        ).clone();

    ncnn::Mat mask_flat(
        static_cast<int>(kMaskCount),
        static_cast<void*>(mask_values.data())
    );

    ncnn::Mat attention_mask =
        mask_flat.reshape(
            kSequenceLength,
            kSequenceLength,
            1
        ).clone();

    ncnn::Mat rope_cos_flat(
        static_cast<int>(kRopeCount),
        static_cast<void*>(rope_cos_values.data())
    );

    // PyTorch shape [4, 1, 313, 128] uses axis 1 as batch.
    // Removing the batch axis gives ncnn [c=4, h=313, w=128].
    ncnn::Mat rope_cos =
        rope_cos_flat.reshape(
            kHeadDim,
            kSequenceLength,
            kRopeComponents
        ).clone();

    ncnn::Mat rope_sin_flat(
        static_cast<int>(kRopeCount),
        static_cast<void*>(rope_sin_values.data())
    );

    ncnn::Mat rope_sin =
        rope_sin_flat.reshape(
            kHeadDim,
            kSequenceLength,
            kRopeComponents
        ).clone();

    std::cout
        << "===== Input shapes =====\n";

    print_shape(
        "in0 hidden_states",
        hidden_states
    );

    print_shape(
        "in1 attention_mask",
        attention_mask
    );

    print_shape(
        "in2 rope_cos",
        rope_cos
    );

    print_shape(
        "in3 rope_sin",
        rope_sin
    );

    if (!is_valid_mat(
            "hidden_states",
            hidden_states,
            kHiddenCount
        ) ||
        !is_valid_mat(
            "attention_mask",
            attention_mask,
            kMaskCount
        ) ||
        !is_valid_mat(
            "rope_cos",
            rope_cos,
            kRopeCount
        ) ||
        !is_valid_mat(
            "rope_sin",
            rope_sin,
            kRopeCount
        )) {
        return EXIT_FAILURE;
    }

    ncnn::Net network;

    network.opt.use_vulkan_compute = false;
    network.opt.use_packing_layout = true;
    network.opt.num_threads = 9;

    const int param_result =
        network.load_param(
            param_path.c_str()
        );

    if (param_result != 0) {
        std::cerr
            << "加载param失败，返回值："
            << param_result
            << '\n';

        return EXIT_FAILURE;
    }

    const int model_result =
        network.load_model(
            model_path.c_str()
        );

    if (model_result != 0) {
        std::cerr
            << "加载bin失败，返回值："
            << model_result
            << '\n';

        return EXIT_FAILURE;
    }

    ncnn::Extractor extractor =
        network.create_extractor();

    if (extractor.input(
            "in0",
            hidden_states
        ) != 0) {
        std::cerr << "写入in0失败。\n";
        return EXIT_FAILURE;
    }

    if (extractor.input(
            "in1",
            attention_mask
        ) != 0) {
        std::cerr << "写入in1失败。\n";
        return EXIT_FAILURE;
    }

    if (extractor.input(
            "in2",
            rope_cos
        ) != 0) {
        std::cerr << "写入in2失败。\n";
        return EXIT_FAILURE;
    }

    if (extractor.input(
            "in3",
            rope_sin
        ) != 0) {
        std::cerr << "写入in3失败。\n";
        return EXIT_FAILURE;
    }

    ncnn::Mat output;

    const int extract_result =
        extractor.extract(
            "out0",
            output
        );

    if (extract_result != 0) {
        std::cerr
            << "提取out0失败，返回值："
            << extract_result
            << '\n';

        return EXIT_FAILURE;
    }

    if (output.empty()) {
        std::cerr << "out0为空。\n";
        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Output shape =====\n";

    print_shape("out0", output);

    if (logical_count(output) != kHiddenCount) {
        std::cerr
            << "输出元素数错误，实际="
            << logical_count(output)
            << "，预期="
            << kHiddenCount
            << '\n';

        return EXIT_FAILURE;
    }

    /*
     * 拉平成一维，避免依赖输出具体是2D还是3D。
     * 当前诊断使用packing layout；输出比较前统一flatten。
     */
    ncnn::Mat output_flat =
        output.reshape(
            static_cast<int>(kHiddenCount)
        );

    if (output_flat.empty()) {
        std::cerr << "输出flatten失败。\n";
        return EXIT_FAILURE;
    }

    const float* actual_values =
        output_flat;

    float maximum_absolute_error = 0.0f;
    double absolute_error_sum = 0.0;
    double squared_error_sum = 0.0;

    double dot_product = 0.0;
    double actual_norm_square = 0.0;
    double expected_norm_square = 0.0;

    std::size_t maximum_error_index = 0;

    for (std::size_t index = 0;
         index < kHiddenCount;
         ++index) {
        const float actual =
            actual_values[index];

        const float expected =
            expected_values[index];

        if (!std::isfinite(actual)) {
            std::cerr
                << "发现非有限输出，index="
                << index
                << '\n';

            return EXIT_FAILURE;
        }

        const float error =
            std::fabs(actual - expected);

        if (error > maximum_absolute_error) {
            maximum_absolute_error = error;
            maximum_error_index = index;
        }

        absolute_error_sum += error;

        squared_error_sum +=
            static_cast<double>(error) *
            static_cast<double>(error);

        dot_product +=
            static_cast<double>(actual) *
            static_cast<double>(expected);

        actual_norm_square +=
            static_cast<double>(actual) *
            static_cast<double>(actual);

        expected_norm_square +=
            static_cast<double>(expected) *
            static_cast<double>(expected);
    }

    const double mean_absolute_error =
        absolute_error_sum /
        static_cast<double>(kHiddenCount);

    const double root_mean_square_error =
        std::sqrt(
            squared_error_sum /
            static_cast<double>(kHiddenCount)
        );

    const double cosine_similarity =
        dot_product /
        (
            std::sqrt(actual_norm_square) *
            std::sqrt(expected_norm_square)
        );

    const std::size_t maximum_error_token =
        maximum_error_index /
        static_cast<std::size_t>(kHiddenSize);

    const std::size_t maximum_error_hidden =
        maximum_error_index %
        static_cast<std::size_t>(kHiddenSize);

    std::cout
        << std::fixed
        << std::setprecision(10);

    std::cout
        << "\n===== Numerical parity =====\n";

    std::cout
        << "Maximum abs error : "
        << maximum_absolute_error
        << '\n';

    std::cout
        << "Mean abs error    : "
        << mean_absolute_error
        << '\n';

    std::cout
        << "RMSE              : "
        << root_mean_square_error
        << '\n';

    std::cout
        << "Cosine similarity : "
        << cosine_similarity
        << '\n';

    std::cout
        << "Max error token   : "
        << maximum_error_token
        << '\n';

    std::cout
        << "Max error hidden  : "
        << maximum_error_hidden
        << '\n';

    std::cout
        << "Actual at max     : "
        << actual_values[maximum_error_index]
        << '\n';

    std::cout
        << "Expected at max   : "
        << expected_values[maximum_error_index]
        << '\n';

    /*
     * Decoder包含多次GEMM、Softmax、RMSNorm和残差运算，
     * CPU实现的浮点累加顺序可能不同，因此不要求逐位完全相同。
     */
    constexpr float kMaximumErrorTolerance =
        0.005f;

    constexpr double kMeanErrorTolerance =
        0.00005;

    constexpr double kCosineTolerance =
        0.999999;

    bool success = true;

    if (
        maximum_absolute_error
        > kMaximumErrorTolerance
    ) {
        std::cerr
            << "❌ 最大绝对误差超限。\n";

        success = false;
    }

    if (
        mean_absolute_error
        > kMeanErrorTolerance
    ) {
        std::cerr
            << "❌ 平均绝对误差超限。\n";

        success = false;
    }

    if (
        cosine_similarity
        < kCosineTolerance
    ) {
        std::cerr
            << "❌ 余弦相似度不足。\n";

        success = false;
    }

    if (!success) {
        return EXIT_FAILURE;
    }

    std::cout
        << "✅ Decoder Layer 0 Prefill "
        << "PyTorch → pnnx → ncnn C++ "
        << "数值对齐成功。\n";

    return EXIT_SUCCESS;
}
