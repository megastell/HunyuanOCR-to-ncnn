#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <net.h>

namespace {

constexpr std::size_t kHiddenSize = 1024;
constexpr std::size_t kVocabSize = 120818;
constexpr int kExpectedToken = 93892;

bool load_float_file(
    const std::string& path,
    const std::size_t expected_count,
    std::vector<float>& values)
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

    const std::streamsize byte_count =
        file.tellg();

    const std::streamsize expected_bytes =
        static_cast<std::streamsize>(
            expected_count * sizeof(float)
        );

    if (byte_count != expected_bytes) {
        std::cerr
            << "文件大小错误："
            << path
            << "\n实际字节数："
            << byte_count
            << "\n预期字节数："
            << expected_bytes
            << '\n';

        return false;
    }

    file.seekg(0, std::ios::beg);

    values.resize(expected_count);

    if (!file.read(
            reinterpret_cast<char*>(
                values.data()
            ),
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

std::size_t logical_element_count(
    const ncnn::Mat& tensor)
{
    return
        static_cast<std::size_t>(tensor.w) *
        static_cast<std::size_t>(tensor.h) *
        static_cast<std::size_t>(tensor.d) *
        static_cast<std::size_t>(tensor.c) *
        static_cast<std::size_t>(
            tensor.elempack
        );
}

int argmax(
    const float* values,
    const std::size_t count)
{
    return static_cast<int>(
        std::distance(
            values,
            std::max_element(
                values,
                values + count
            )
        )
    );
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <lm_head.ncnn.param>"
            << " <lm_head.ncnn.bin>"
            << " <lm_head_input_f32.bin>"
            << " <lm_head_logits_f32.bin>\n";

        return EXIT_FAILURE;
    }

    const std::string param_path = argv[1];
    const std::string model_path = argv[2];
    const std::string input_path = argv[3];
    const std::string expected_path = argv[4];

    std::vector<float> input_values;
    std::vector<float> expected_values;

    if (!load_float_file(
            input_path,
            kHiddenSize,
            input_values
        )) {
        return EXIT_FAILURE;
    }

    if (!load_float_file(
            expected_path,
            kVocabSize,
            expected_values
        )) {
        return EXIT_FAILURE;
    }

    ncnn::Net network;

    network.opt.use_vulkan_compute = false;
    network.opt.use_packing_layout = false;
    network.opt.num_threads = 9;

    const int param_result =
        network.load_param(
            param_path.c_str()
        );

    if (param_result != 0) {
        std::cerr
            << "加载 param 失败，返回值："
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
            << "加载 bin 失败，返回值："
            << model_result
            << '\n';

        return EXIT_FAILURE;
    }

    ncnn::Mat input_tensor(
        static_cast<int>(kHiddenSize)
    );

    std::memcpy(
        input_tensor.data,
        input_values.data(),
        kHiddenSize * sizeof(float)
    );

    ncnn::Extractor extractor =
        network.create_extractor();

    const int input_result =
        extractor.input(
            "in0",
            input_tensor
        );

    if (input_result != 0) {
        std::cerr
            << "写入 in0 失败，返回值："
            << input_result
            << '\n';

        return EXIT_FAILURE;
    }

    ncnn::Mat output_tensor;

    const int output_result =
        extractor.extract(
            "out0",
            output_tensor
        );

    if (output_result != 0) {
        std::cerr
            << "读取 out0 失败，返回值："
            << output_result
            << '\n';

        return EXIT_FAILURE;
    }

    if (output_tensor.empty()) {
        std::cerr << "ncnn 输出为空。\n";
        return EXIT_FAILURE;
    }

    const std::size_t actual_count =
        logical_element_count(
            output_tensor
        );

    std::cout
        << "ncnn output shape: "
        << "dims=" << output_tensor.dims
        << ", w=" << output_tensor.w
        << ", h=" << output_tensor.h
        << ", d=" << output_tensor.d
        << ", c=" << output_tensor.c
        << ", elempack="
        << output_tensor.elempack
        << ", logical_count="
        << actual_count
        << ", storage_total="
        << output_tensor.total()
        << '\n';

    if (actual_count != kVocabSize) {
        std::cerr
            << "输出元素数量错误。ncnn="
            << actual_count
            << "，预期="
            << kVocabSize
            << '\n';

        return EXIT_FAILURE;
    }

    const float* actual_values =
        output_tensor;

    double sum_absolute_error = 0.0;
    double dot_product = 0.0;
    double actual_norm_square = 0.0;
    double expected_norm_square = 0.0;
    float max_absolute_error = 0.0f;

    for (std::size_t index = 0;
         index < kVocabSize;
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

        max_absolute_error =
            std::max(
                max_absolute_error,
                error
            );

        sum_absolute_error += error;

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
        sum_absolute_error /
        static_cast<double>(kVocabSize);

    const double cosine_similarity =
        dot_product /
        (
            std::sqrt(actual_norm_square) *
            std::sqrt(expected_norm_square)
        );

    const int expected_token =
        argmax(
            expected_values.data(),
            kVocabSize
        );

    const int actual_token =
        argmax(
            actual_values,
            kVocabSize
        );

    std::cout
        << std::fixed
        << std::setprecision(10);

    std::cout
        << "Expected argmax token: "
        << expected_token
        << '\n';

    std::cout
        << "ncnn argmax token    : "
        << actual_token
        << '\n';

    std::cout
        << "Maximum abs error    : "
        << max_absolute_error
        << '\n';

    std::cout
        << "Mean abs error       : "
        << mean_absolute_error
        << '\n';

    std::cout
        << "Cosine similarity    : "
        << cosine_similarity
        << '\n';

    constexpr float kMaxErrorTolerance =
        0.003f;

    constexpr double kMeanErrorTolerance =
        0.00005;

    constexpr double kCosineTolerance =
        0.999999;

    bool success = true;

    if (expected_token != kExpectedToken) {
        std::cerr
            << "参考 logits 的 argmax "
            << "不是拆分契约中的 93892。\n";

        success = false;
    }

    if (actual_token != kExpectedToken) {
        std::cerr
            << "❌ ncnn 首 token 不一致。\n";

        success = false;
    }

    if (
        max_absolute_error
        > kMaxErrorTolerance
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
        << "✅ LM Head PyTorch → pnnx → "
        << "ncnn C++ CPU 严格对齐成功。\n";

    return EXIT_SUCCESS;
}
