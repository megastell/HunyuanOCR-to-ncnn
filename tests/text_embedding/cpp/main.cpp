#include <algorithm>
#include <cmath>
#include <cstdint>
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
constexpr std::size_t kOutputCount =
    kSequenceLength * kHiddenSize;

template <typename T>
bool load_binary(
    const std::string& path,
    std::size_t expected_count,
    std::vector<T>& output)
{
    std::ifstream file(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!file) {
        std::cerr << "无法打开文件：" << path << '\n';
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
            << "文件大小错误：" << path
            << "\n实际字节数：" << actual_bytes
            << "\n预期字节数：" << expected_bytes
            << '\n';
        return false;
    }

    file.seekg(0, std::ios::beg);
    output.resize(expected_count);

    if (!file.read(
            reinterpret_cast<char*>(output.data()),
            expected_bytes
        )) {
        std::cerr << "读取文件失败：" << path << '\n';
        return false;
    }

    return true;
}

std::size_t logical_count(const ncnn::Mat& tensor)
{
    return
        static_cast<std::size_t>(tensor.w) *
        static_cast<std::size_t>(tensor.h) *
        static_cast<std::size_t>(tensor.d) *
        static_cast<std::size_t>(tensor.c) *
        static_cast<std::size_t>(tensor.elempack);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <text_embedding.ncnn.param>"
            << " <text_embedding.ncnn.bin>"
            << " <input_ids_i32.bin>"
            << " <expected_output_f32.bin>\n";

        return EXIT_FAILURE;
    }

    std::vector<std::int32_t> input_ids;
    std::vector<float> expected;

    if (!load_binary(
            argv[3],
            kSequenceLength,
            input_ids
        )) {
        return EXIT_FAILURE;
    }

    if (!load_binary(
            argv[4],
            kOutputCount,
            expected
        )) {
        return EXIT_FAILURE;
    }

    ncnn::Net network;
    network.opt.use_vulkan_compute = false;
    network.opt.use_packing_layout = false;
    network.opt.num_threads = 9;

    int result = network.load_param(argv[1]);

    if (result != 0) {
        std::cerr
            << "加载param失败，返回值："
            << result << '\n';
        return EXIT_FAILURE;
    }

    result = network.load_model(argv[2]);

    if (result != 0) {
        std::cerr
            << "加载bin失败，返回值："
            << result << '\n';
        return EXIT_FAILURE;
    }

    // ncnn Embed层要求输入为int32 token ID。
    ncnn::Mat input(
        static_cast<int>(kSequenceLength),
        sizeof(std::int32_t)
    );

    std::memcpy(
        input.data,
        input_ids.data(),
        kSequenceLength * sizeof(std::int32_t)
    );

    ncnn::Extractor extractor =
        network.create_extractor();

    result = extractor.input("in0", input);

    if (result != 0) {
        std::cerr
            << "写入in0失败，返回值："
            << result << '\n';
        return EXIT_FAILURE;
    }

    ncnn::Mat output;
    result = extractor.extract("out0", output);

    if (result != 0 || output.empty()) {
        std::cerr
            << "读取out0失败，返回值："
            << result << '\n';
        return EXIT_FAILURE;
    }

    const std::size_t actual_count =
        logical_count(output);

    std::cout
        << "ncnn output shape: "
        << "dims=" << output.dims
        << ", w=" << output.w
        << ", h=" << output.h
        << ", d=" << output.d
        << ", c=" << output.c
        << ", elempack=" << output.elempack
        << ", logical_count=" << actual_count
        << '\n';

    if (
        output.w != static_cast<int>(kHiddenSize) ||
        output.h != static_cast<int>(kSequenceLength) ||
        actual_count != kOutputCount
    ) {
        std::cerr
            << "❌ ncnn输出形状不符合契约。\n";
        return EXIT_FAILURE;
    }

    float max_error = 0.0f;
    double error_sum = 0.0;
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;

    for (std::size_t token = 0;
         token < kSequenceLength;
         ++token) {
        const float* actual_row =
            output.row(static_cast<int>(token));

        const float* expected_row =
            expected.data() + token * kHiddenSize;

        for (std::size_t hidden = 0;
             hidden < kHiddenSize;
             ++hidden) {
            const float actual_value =
                actual_row[hidden];

            const float expected_value =
                expected_row[hidden];

            if (!std::isfinite(actual_value)) {
                std::cerr
                    << "发现非有限输出：token="
                    << token
                    << " hidden="
                    << hidden
                    << '\n';
                return EXIT_FAILURE;
            }

            const float error =
                std::fabs(
                    actual_value - expected_value
                );

            max_error = std::max(max_error, error);
            error_sum += error;

            dot +=
                static_cast<double>(actual_value) *
                static_cast<double>(expected_value);

            actual_norm +=
                static_cast<double>(actual_value) *
                static_cast<double>(actual_value);

            expected_norm +=
                static_cast<double>(expected_value) *
                static_cast<double>(expected_value);
        }
    }

    const double mean_error =
        error_sum /
        static_cast<double>(kOutputCount);

    const double cosine =
        dot /
        (
            std::sqrt(actual_norm) *
            std::sqrt(expected_norm)
        );

    std::cout
        << std::fixed
        << std::setprecision(10);

    std::cout
        << "Maximum abs error : "
        << max_error << '\n';

    std::cout
        << "Mean abs error    : "
        << mean_error << '\n';

    std::cout
        << "Cosine similarity : "
        << cosine << '\n';

    constexpr float kMaxTolerance = 0.000001f;
    constexpr double kMeanTolerance = 0.00000001;
    constexpr double kCosineTolerance = 0.999999999;

    if (
        max_error > kMaxTolerance ||
        mean_error > kMeanTolerance ||
        cosine < kCosineTolerance
    ) {
        std::cerr
            << "❌ Text Embedding数值对齐失败。\n";
        return EXIT_FAILURE;
    }

    std::cout
        << "✅ Text Embedding PyTorch → "
        << "pnnx → ncnn C++严格对齐成功。\n";

    return EXIT_SUCCESS;
}
