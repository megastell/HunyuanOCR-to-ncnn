#include <net.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kTokenId = 5112;
constexpr std::size_t kHiddenSize = 1024;
constexpr int kThreads = 9;

bool load_exact_f32(
    const std::string& path,
    std::size_t expected_count,
    std::vector<float>& values)
{
    std::ifstream stream(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!stream) {
        std::cerr
            << "无法打开参考文件："
            << path
            << '\n';

        return false;
    }

    const std::streamsize file_size =
        stream.tellg();

    const std::streamsize expected_size =
        static_cast<std::streamsize>(
            expected_count * sizeof(float)
        );

    if (file_size != expected_size) {
        std::cerr
            << "参考文件大小错误："
            << path
            << "\n实际："
            << file_size
            << "，预期："
            << expected_size
            << '\n';

        return false;
    }

    stream.seekg(
        0,
        std::ios::beg
    );

    values.resize(
        expected_count
    );

    if (
        !stream.read(
            reinterpret_cast<char*>(
                values.data()
            ),
            expected_size
        )
    ) {
        std::cerr
            << "参考文件读取失败："
            << path
            << '\n';

        return false;
    }

    return true;
}

std::size_t logical_count(
    const ncnn::Mat& matrix)
{
    std::size_t count = 0;

    if (matrix.dims == 1) {
        count =
            static_cast<std::size_t>(
                matrix.w
            );
    } else if (matrix.dims == 2) {
        count =
            static_cast<std::size_t>(
                matrix.w
            )
            * static_cast<std::size_t>(
                matrix.h
            );
    } else if (matrix.dims == 3) {
        count =
            static_cast<std::size_t>(
                matrix.w
            )
            * static_cast<std::size_t>(
                matrix.h
            )
            * static_cast<std::size_t>(
                matrix.c
            );
    } else if (matrix.dims == 4) {
        count =
            static_cast<std::size_t>(
                matrix.w
            )
            * static_cast<std::size_t>(
                matrix.h
            )
            * static_cast<std::size_t>(
                matrix.d
            )
            * static_cast<std::size_t>(
                matrix.c
            );
    }

    return
        count
        * static_cast<std::size_t>(
            matrix.elempack
        );
}

void print_shape(
    const ncnn::Mat& matrix)
{
    std::cout
        << "dims="
        << matrix.dims
        << ", w="
        << matrix.w
        << ", h="
        << matrix.h
        << ", d="
        << matrix.d
        << ", c="
        << matrix.c
        << ", elempack="
        << matrix.elempack
        << ", elemsize="
        << matrix.elemsize
        << ", logical_count="
        << logical_count(matrix)
        << '\n';
}

}  // namespace

int main(
    int argc,
    char** argv)
{
    if (argc != 6) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <param>"
            << " <bin>"
            << " <expected_f32>"
            << " <packing:0|1>"
            << " <token_id>\n";

        return EXIT_FAILURE;
    }

    const std::string param_path =
        argv[1];

    const std::string model_path =
        argv[2];

    const std::string expected_path =
        argv[3];

    const std::string packing_text =
        argv[4];

    const int token_id =
        std::stoi(
            argv[5]
        );

    if (
        packing_text != "0"
        && packing_text != "1"
    ) {
        std::cerr
            << "packing参数只能为0或1。\n";

        return EXIT_FAILURE;
    }

    if (token_id != kTokenId) {
        std::cerr
            << "本阶段参考输出只对应token "
            << kTokenId
            << "，实际输入为"
            << token_id
            << "。\n";

        return EXIT_FAILURE;
    }

    const bool use_packing_layout =
        packing_text == "1";

    std::vector<float> expected;

    if (
        !load_exact_f32(
            expected_path,
            kHiddenSize,
            expected
        )
    ) {
        return EXIT_FAILURE;
    }

    ncnn::Net network;

    network.opt.use_vulkan_compute =
        false;

    network.opt.use_packing_layout =
        use_packing_layout;

    network.opt.num_threads =
        kThreads;

    if (
        network.load_param(
            param_path.c_str()
        ) != 0
    ) {
        std::cerr
            << "ncnn param加载失败："
            << param_path
            << '\n';

        return EXIT_FAILURE;
    }

    if (
        network.load_model(
            model_path.c_str()
        ) != 0
    ) {
        std::cerr
            << "ncnn bin加载失败："
            << model_path
            << '\n';

        return EXIT_FAILURE;
    }

    /*
     * ncnn Embed::forward()将bottom_blob按const int*读取。
     * 因此这里必须创建4字节int32 token Mat，
     * 不能把5112作为float传入。
     */
    ncnn::Mat token_input(
        1,
        static_cast<std::size_t>(4u)
    );

    if (token_input.empty()) {
        std::cerr
            << "token输入Mat创建失败。\n";

        return EXIT_FAILURE;
    }

    int* token_pointer =
        token_input;

    token_pointer[0] =
        token_id;

    ncnn::Extractor extractor =
        network.create_extractor();

    extractor.set_light_mode(
        false
    );

    if (
        extractor.input(
            "in0",
            token_input
        ) != 0
    ) {
        std::cerr
            << "Embedding输入绑定失败。\n";

        return EXIT_FAILURE;
    }

    ncnn::Mat output;

    if (
        extractor.extract(
            "out0",
            output
        ) != 0
        || output.empty()
    ) {
        std::cerr
            << "Embedding输出提取失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "===== Token Embedding runtime =====\n"
        << "token ID       : "
        << token_id
        << '\n'
        << "threads        : "
        << kThreads
        << '\n'
        << "packing layout : "
        << (
            use_packing_layout
            ? "true"
            : "false"
        )
        << '\n'
        << "input type     : int32\n"
        << "input shape    : ";

    print_shape(
        token_input
    );

    std::cout
        << "output shape   : ";

    print_shape(
        output
    );

    const std::size_t output_count =
        logical_count(
            output
        );

    if (output_count != kHiddenSize) {
        std::cerr
            << "Embedding输出元素数量错误："
            << output_count
            << "，预期："
            << kHiddenSize
            << '\n';

        return EXIT_FAILURE;
    }

    if (
        output.elemsize
        / static_cast<std::size_t>(
            output.elempack
        )
        != sizeof(float)
    ) {
        std::cerr
            << "Embedding输出不是FP32。\n";

        return EXIT_FAILURE;
    }

    const float* actual_pointer =
        output;

    double maximum_abs_error = 0.0;
    double total_abs_error = 0.0;
    double squared_error = 0.0;

    std::size_t maximum_error_index = 0;

    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;

    for (
        std::size_t index = 0;
        index < kHiddenSize;
        ++index
    ) {
        const double actual =
            static_cast<double>(
                actual_pointer[index]
            );

        const double expected_value =
            static_cast<double>(
                expected[index]
            );

        const double difference =
            actual
            - expected_value;

        const double absolute_error =
            std::fabs(
                difference
            );

        if (
            absolute_error
            > maximum_abs_error
        ) {
            maximum_abs_error =
                absolute_error;

            maximum_error_index =
                index;
        }

        total_abs_error +=
            absolute_error;

        squared_error +=
            difference * difference;

        dot +=
            actual * expected_value;

        actual_norm +=
            actual * actual;

        expected_norm +=
            expected_value
            * expected_value;
    }

    const double mean_abs_error =
        total_abs_error
        / static_cast<double>(
            kHiddenSize
        );

    const double rmse =
        std::sqrt(
            squared_error
            / static_cast<double>(
                kHiddenSize
            )
        );

    const double cosine_similarity =
        dot
        / (
            std::sqrt(actual_norm)
            * std::sqrt(expected_norm)
        );

    const bool byte_identical =
        std::memcmp(
            actual_pointer,
            expected.data(),
            kHiddenSize * sizeof(float)
        )
        == 0;

    std::cout
        << std::scientific
        << std::setprecision(10)
        << "\n===== Numerical parity =====\n"
        << "Maximum abs error : "
        << maximum_abs_error
        << '\n'
        << "Mean abs error    : "
        << mean_abs_error
        << '\n'
        << "RMSE              : "
        << rmse
        << '\n'
        << std::fixed
        << std::setprecision(12)
        << "Cosine similarity : "
        << cosine_similarity
        << '\n'
        << "Maximum error idx : "
        << maximum_error_index
        << '\n'
        << "Byte-identical    : "
        << (
            byte_identical
            ? "true"
            : "false"
        )
        << '\n';

    if (
        maximum_abs_error != 0.0
        || mean_abs_error != 0.0
        || !byte_identical
    ) {
        std::cerr
            << "❌ token 5112 ncnn Embedding"
            << "未达到字节级一致。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "\n✅ token 5112"
        << " → ncnn Embed"
        << " → Step 2 Layer 0 hidden"
        << " 字节级一致。\n";

    return EXIT_SUCCESS;
}
