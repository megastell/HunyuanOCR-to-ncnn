#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <net.h>

namespace {

constexpr std::size_t kSequenceLength = 313;
constexpr std::size_t kHiddenSize = 1024;
constexpr std::size_t kElementCount =
    kSequenceLength * kHiddenSize;

bool load_f32(
    const std::string& path,
    std::vector<float>& values)
{
    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!file) {
        std::cerr << "无法打开：" << path << '\n';
        return false;
    }

    const std::streamsize expected_bytes =
        static_cast<std::streamsize>(
            kElementCount * sizeof(float)
        );

    const std::streamsize actual_bytes = file.tellg();

    if (actual_bytes != expected_bytes) {
        std::cerr
            << "文件大小错误：" << path
            << "\n实际：" << actual_bytes
            << "\n预期：" << expected_bytes
            << '\n';
        return false;
    }

    file.seekg(0, std::ios::beg);
    values.resize(kElementCount);

    return static_cast<bool>(
        file.read(
            reinterpret_cast<char*>(values.data()),
            expected_bytes
        )
    );
}

std::size_t logical_count(const ncnn::Mat& value)
{
    return
        static_cast<std::size_t>(value.w) *
        static_cast<std::size_t>(value.h) *
        static_cast<std::size_t>(value.d) *
        static_cast<std::size_t>(value.c) *
        static_cast<std::size_t>(value.elempack);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr
            << "用法："
            << argv[0]
            << " <param> <bin> <input_f32> <expected_f32>\n";
        return EXIT_FAILURE;
    }

    std::vector<float> input_values;
    std::vector<float> expected_values;

    if (!load_f32(argv[3], input_values) ||
        !load_f32(argv[4], expected_values)) {
        return EXIT_FAILURE;
    }

    ncnn::Net network;
    network.opt.use_vulkan_compute = false;
    network.opt.use_packing_layout = false;
    network.opt.num_threads = 9;

    if (network.load_param(argv[1]) != 0) {
        std::cerr << "加载param失败。\n";
        return EXIT_FAILURE;
    }

    if (network.load_model(argv[2]) != 0) {
        std::cerr << "加载bin失败。\n";
        return EXIT_FAILURE;
    }

    ncnn::Mat input(
        static_cast<int>(kHiddenSize),
        static_cast<int>(kSequenceLength)
    );

    std::memcpy(
        input.data,
        input_values.data(),
        kElementCount * sizeof(float)
    );

    ncnn::Extractor extractor = network.create_extractor();

    if (extractor.input("in0", input) != 0) {
        std::cerr << "输入in0失败。\n";
        return EXIT_FAILURE;
    }

    ncnn::Mat output;

    if (extractor.extract("out0", output) != 0 ||
        output.empty()) {
        std::cerr << "提取out0失败。\n";
        return EXIT_FAILURE;
    }

    const std::size_t count = logical_count(output);

    std::cout
        << "ncnn output shape: "
        << "dims=" << output.dims
        << ", w=" << output.w
        << ", h=" << output.h
        << ", d=" << output.d
        << ", c=" << output.c
        << ", elempack=" << output.elempack
        << ", logical_count=" << count
        << '\n';

    if (output.w != static_cast<int>(kHiddenSize) ||
        output.h != static_cast<int>(kSequenceLength) ||
        count != kElementCount) {
        std::cerr << "❌ 输出形状不正确。\n";
        return EXIT_FAILURE;
    }

    float max_error = 0.0f;
    double error_sum = 0.0;
    double dot = 0.0;
    double actual_square_sum = 0.0;
    double expected_square_sum = 0.0;

    for (std::size_t row = 0;
         row < kSequenceLength;
         ++row) {
        const float* actual =
            output.row(static_cast<int>(row));

        const float* expected =
            expected_values.data() + row * kHiddenSize;

        for (std::size_t column = 0;
             column < kHiddenSize;
             ++column) {
            const float actual_value = actual[column];
            const float expected_value = expected[column];

            if (!std::isfinite(actual_value)) {
                std::cerr << "发现非有限值。\n";
                return EXIT_FAILURE;
            }

            const float error =
                std::fabs(actual_value - expected_value);

            max_error = std::max(max_error, error);
            error_sum += error;

            dot +=
                static_cast<double>(actual_value) *
                static_cast<double>(expected_value);

            actual_square_sum +=
                static_cast<double>(actual_value) *
                static_cast<double>(actual_value);

            expected_square_sum +=
                static_cast<double>(expected_value) *
                static_cast<double>(expected_value);
        }
    }

    const double mean_error =
        error_sum / static_cast<double>(kElementCount);

    const double cosine =
        dot /
        (
            std::sqrt(actual_square_sum) *
            std::sqrt(expected_square_sum)
        );

    std::cout
        << std::fixed
        << std::setprecision(10)
        << "Maximum abs error : " << max_error << '\n'
        << "Mean abs error    : " << mean_error << '\n'
        << "Cosine similarity : " << cosine << '\n';

    constexpr float kMaximumTolerance = 0.00002f;
    constexpr double kMeanTolerance = 0.000001;
    constexpr double kCosineTolerance = 0.99999999;

    if (max_error > kMaximumTolerance ||
        mean_error > kMeanTolerance ||
        cosine < kCosineTolerance) {
        std::cerr << "❌ Final RMSNorm数值对齐失败。\n";
        return EXIT_FAILURE;
    }

    std::cout
        << "✅ Final RMSNorm PyTorch → pnnx → "
        << "ncnn C++严格对齐成功。\n";

    return EXIT_SUCCESS;
}
