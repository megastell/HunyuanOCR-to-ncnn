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

constexpr int kHiddenSize = 1024;
constexpr int kMaskLength = 314;

constexpr int kRopeComponents = 4;
constexpr int kHeadDim = 128;

constexpr int kKeyValueHeads = 8;
constexpr int kPastLength = 313;
constexpr int kPresentLength = 314;

constexpr int kThreads = 9;

constexpr std::size_t kHiddenCount =
    static_cast<std::size_t>(kHiddenSize);

constexpr std::size_t kMaskCount =
    static_cast<std::size_t>(kMaskLength);

constexpr std::size_t kRopeCount =
    static_cast<std::size_t>(kRopeComponents)
    * static_cast<std::size_t>(kHeadDim);

constexpr std::size_t kPastCacheCount =
    static_cast<std::size_t>(kKeyValueHeads)
    * static_cast<std::size_t>(kPastLength)
    * static_cast<std::size_t>(kHeadDim);

constexpr std::size_t kPresentCacheCount =
    static_cast<std::size_t>(kKeyValueHeads)
    * static_cast<std::size_t>(kPresentLength)
    * static_cast<std::size_t>(kHeadDim);


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
            << "文件大小不正确："
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


std::size_t logical_count(
    const ncnn::Mat& value)
{
    return
        static_cast<std::size_t>(value.w)
        * static_cast<std::size_t>(value.h)
        * static_cast<std::size_t>(value.d)
        * static_cast<std::size_t>(value.c)
        * static_cast<std::size_t>(
            value.elempack
        );
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


/*
 * 将ncnn Mat转换回逻辑顺序的一维数组。
 *
 * dims=1:
 *   logical shape = [w * elempack]
 *
 * dims=2:
 *   logical shape = [h * elempack, w]
 *
 * dims=3:
 *   logical shape = [c * elempack, h, w]
 *
 * Decode输出预计分别为：
 *   out0: dims=1
 *   out1: dims=3
 *   out2: dims=3
 */
bool unpack_mat(
    const ncnn::Mat& value,
    std::vector<float>& result)
{
    if (value.empty()) {
        std::cerr
            << "无法展开空Mat。\n";

        return false;
    }

    if (
        value.elemsize
        != sizeof(float)
           * static_cast<std::size_t>(
               value.elempack
           )
    ) {
        std::cerr
            << "当前程序只支持FP32 Mat，"
            << "elemsize="
            << value.elemsize
            << "，elempack="
            << value.elempack
            << '\n';

        return false;
    }

    result.assign(
        logical_count(value),
        0.0f
    );

    const int pack = value.elempack;

    if (value.dims == 1) {
        const float* source = value;

        for (int x = 0; x < value.w; ++x) {
            for (int p = 0; p < pack; ++p) {
                const std::size_t destination_index =
                    static_cast<std::size_t>(x)
                    * static_cast<std::size_t>(pack)
                    + static_cast<std::size_t>(p);

                const std::size_t source_index =
                    destination_index;

                result[destination_index] =
                    source[source_index];
            }
        }

        return true;
    }

    if (value.dims == 2) {
        for (int y = 0; y < value.h; ++y) {
            const float* source =
                value.row(y);

            for (int x = 0; x < value.w; ++x) {
                for (int p = 0; p < pack; ++p) {
                    const int logical_y =
                        y * pack + p;

                    const std::size_t destination_index =
                        static_cast<std::size_t>(
                            logical_y
                        )
                        * static_cast<std::size_t>(
                            value.w
                        )
                        + static_cast<std::size_t>(x);

                    const std::size_t source_index =
                        static_cast<std::size_t>(x)
                        * static_cast<std::size_t>(pack)
                        + static_cast<std::size_t>(p);

                    result[destination_index] =
                        source[source_index];
                }
            }
        }

        return true;
    }

    if (value.dims == 3) {
        for (int q = 0; q < value.c; ++q) {
            const float* source =
                value.channel(q);

            for (int y = 0; y < value.h; ++y) {
                for (int x = 0; x < value.w; ++x) {
                    for (int p = 0; p < pack; ++p) {
                        const int logical_channel =
                            q * pack + p;

                        const std::size_t destination_index =
                            (
                                (
                                    static_cast<std::size_t>(
                                        logical_channel
                                    )
                                    * static_cast<std::size_t>(
                                        value.h
                                    )
                                    + static_cast<std::size_t>(
                                        y
                                    )
                                )
                                * static_cast<std::size_t>(
                                    value.w
                                )
                                + static_cast<std::size_t>(
                                    x
                                )
                            );

                        const std::size_t source_index =
                            (
                                (
                                    static_cast<std::size_t>(
                                        y
                                    )
                                    * static_cast<std::size_t>(
                                        value.w
                                    )
                                    + static_cast<std::size_t>(
                                        x
                                    )
                                )
                                * static_cast<std::size_t>(
                                    pack
                                )
                                + static_cast<std::size_t>(
                                    p
                                )
                            );

                        result[destination_index] =
                            source[source_index];
                    }
                }
            }
        }

        return true;
    }

    std::cerr
        << "当前不支持dims="
        << value.dims
        << "的输出Mat。\n";

    return false;
}


struct Metrics {
    double maximum_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double rmse = 0.0;
    double cosine_similarity = 0.0;

    std::size_t max_error_index = 0;

    double actual_at_max = 0.0;
    double expected_at_max = 0.0;
};


bool calculate_metrics(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    Metrics& metrics)
{
    if (actual.size() != expected.size()) {
        std::cerr
            << "比较数组大小不同：actual="
            << actual.size()
            << "，expected="
            << expected.size()
            << '\n';

        return false;
    }

    if (actual.empty()) {
        std::cerr
            << "不能比较空数组。\n";

        return false;
    }

    double absolute_sum = 0.0;
    double squared_sum = 0.0;

    double dot_product = 0.0;
    double actual_norm_squared = 0.0;
    double expected_norm_squared = 0.0;

    for (
        std::size_t index = 0;
        index < actual.size();
        ++index
    ) {
        const double actual_value =
            static_cast<double>(actual[index]);

        const double expected_value =
            static_cast<double>(expected[index]);

        if (
            !std::isfinite(actual_value)
            || !std::isfinite(expected_value)
        ) {
            std::cerr
                << "发现非有限数，index="
                << index
                << '\n';

            return false;
        }

        const double difference =
            actual_value - expected_value;

        const double absolute_difference =
            std::abs(difference);

        if (
            absolute_difference
            > metrics.maximum_abs_error
        ) {
            metrics.maximum_abs_error =
                absolute_difference;

            metrics.max_error_index =
                index;

            metrics.actual_at_max =
                actual_value;

            metrics.expected_at_max =
                expected_value;
        }

        absolute_sum +=
            absolute_difference;

        squared_sum +=
            difference * difference;

        dot_product +=
            actual_value * expected_value;

        actual_norm_squared +=
            actual_value * actual_value;

        expected_norm_squared +=
            expected_value * expected_value;
    }

    metrics.mean_abs_error =
        absolute_sum
        / static_cast<double>(actual.size());

    metrics.rmse = std::sqrt(
        squared_sum
        / static_cast<double>(actual.size())
    );

    const double denominator = std::sqrt(
        actual_norm_squared
        * expected_norm_squared
    );

    if (denominator == 0.0) {
        metrics.cosine_similarity =
            actual == expected ? 1.0 : 0.0;
    } else {
        const double raw_cosine =
            dot_product / denominator;

        metrics.cosine_similarity =
            std::clamp(
                raw_cosine,
                -1.0,
                1.0
            );
    }

    return true;
}


void print_metrics(
    const std::string& name,
    const Metrics& metrics)
{
    std::cout
        << '\n'
        << name
        << '\n'
        << std::scientific
        << std::setprecision(10)
        << "  Maximum abs error : "
        << metrics.maximum_abs_error
        << '\n'
        << "  Mean abs error    : "
        << metrics.mean_abs_error
        << '\n'
        << "  RMSE              : "
        << metrics.rmse
        << '\n'
        << std::fixed
        << std::setprecision(12)
        << "  Cosine similarity : "
        << metrics.cosine_similarity
        << '\n'
        << "  Max error index   : "
        << metrics.max_error_index
        << '\n'
        << std::setprecision(10)
        << "  Actual at max     : "
        << metrics.actual_at_max
        << '\n'
        << "  Expected at max   : "
        << metrics.expected_at_max
        << '\n';
}


}  // namespace


int main(
    int argc,
    char** argv)
{
    if (argc != 3) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <project_root>"
            << " <packing:0|1>\n";

        return EXIT_FAILURE;
    }

    const std::string project_root =
        argv[1];

    const std::string packing_text =
        argv[2];

    if (
        packing_text != "0"
        && packing_text != "1"
    ) {
        std::cerr
            << "packing参数只能为0或1。\n";

        return EXIT_FAILURE;
    }

    const bool use_packing_layout =
        packing_text == "1";

    constexpr std::size_t kDecodeVocabSize =
        120818;

    const std::string layer23_reference =
        project_root
        + "/artifacts/decoder_layer23_decode"
        + "/reference";

    const std::string tail_reference =
        project_root
        + "/artifacts/decode_tail/reference";

    const std::string final_norm_directory =
        project_root
        + "/artifacts/final_norm";

    const std::string lm_head_directory =
        project_root
        + "/artifacts/lm_head";

    std::vector<float> layer23_output_values;
    std::vector<float> expected_norm_input;
    std::vector<float> expected_norm_output;
    std::vector<float> expected_logits;

    if (
        !load_exact_binary(
            layer23_reference
                + "/layer23_output_f32.bin",
            kHiddenCount,
            layer23_output_values
        )
        || !load_exact_binary(
            tail_reference
                + "/final_norm_input_f32.bin",
            kHiddenCount,
            expected_norm_input
        )
        || !load_exact_binary(
            tail_reference
                + "/final_norm_output_f32.bin",
            kHiddenCount,
            expected_norm_output
        )
        || !load_exact_binary(
            tail_reference
                + "/decode_logits_f32.bin",
            kDecodeVocabSize,
            expected_logits
        )
    ) {
        return EXIT_FAILURE;
    }

    Metrics input_boundary_metrics;

    if (
        !calculate_metrics(
            layer23_output_values,
            expected_norm_input,
            input_boundary_metrics
        )
    ) {
        return EXIT_FAILURE;
    }

    std::cout
        << "===== Decode-tail runtime options =====\n"
        << "project root        : "
        << project_root
        << '\n'
        << "threads             : "
        << kThreads
        << '\n'
        << "packing layout      : "
        << (
            use_packing_layout
            ? "true"
            : "false"
        )
        << '\n'
        << "intermediate reload : disabled\n"
        << "handoff 1           : "
        << "Layer23 output -> Final RMSNorm in0\n"
        << "handoff 2           : "
        << "Final RMSNorm out0.clone() -> LM Head in0\n";

    std::cout
        << "\n===== Static boundary check =====\n";

    print_metrics(
        "Layer23 output vs Final RMSNorm reference input",
        input_boundary_metrics
    );

    if (
        input_boundary_metrics.maximum_abs_error
        != 0.0
    ) {
        std::cerr
            << "❌ Layer23输出与Final RMSNorm"
            << "参考输入不完全一致。\n";

        return EXIT_FAILURE;
    }

    ncnn::Mat hidden_flat(
        static_cast<int>(kHiddenCount),
        static_cast<void*>(
            layer23_output_values.data()
        )
    );

    ncnn::Mat current_hidden =
        hidden_flat.reshape(
            kHiddenSize,
            1
        ).clone();

    if (current_hidden.empty()) {
        std::cerr
            << "创建Layer23输出Mat失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Final RMSNorm input =====\n";

    print_shape(
        "final_norm_input",
        current_hidden
    );

    Metrics norm_metrics;
    ncnn::Mat norm_handoff;

    {
        ncnn::Net final_norm_network;

        final_norm_network.opt.use_vulkan_compute =
            false;

        final_norm_network.opt.use_packing_layout =
            use_packing_layout;

        final_norm_network.opt.num_threads =
            kThreads;

        const std::string param_path =
            final_norm_directory
            + "/final_norm.ncnn.param";

        const std::string model_path =
            final_norm_directory
            + "/final_norm.ncnn.bin";

        if (
            final_norm_network.load_param(
                param_path.c_str()
            ) != 0
        ) {
            std::cerr
                << "Final RMSNorm param加载失败："
                << param_path
                << '\n';

            return EXIT_FAILURE;
        }

        if (
            final_norm_network.load_model(
                model_path.c_str()
            ) != 0
        ) {
            std::cerr
                << "Final RMSNorm bin加载失败："
                << model_path
                << '\n';

            return EXIT_FAILURE;
        }

        ncnn::Extractor extractor =
            final_norm_network.create_extractor();

        extractor.set_light_mode(false);

        const int input_status =
            extractor.input(
                "in0",
                current_hidden
            );

        if (input_status != 0) {
            std::cerr
                << "Final RMSNorm输入绑定失败："
                << input_status
                << '\n';

            return EXIT_FAILURE;
        }

        ncnn::Mat norm_output;

        const int output_status =
            extractor.extract(
                "out0",
                norm_output
            );

        if (output_status != 0) {
            std::cerr
                << "Final RMSNorm输出提取失败："
                << output_status
                << '\n';

            return EXIT_FAILURE;
        }

        if (norm_output.empty()) {
            std::cerr
                << "Final RMSNorm输出为空。\n";

            return EXIT_FAILURE;
        }

        std::cout
            << "\n===== Final RMSNorm output =====\n";

        print_shape(
            "final_norm_output",
            norm_output
        );

        std::vector<float> actual_norm_output;

        if (
            !unpack_mat(
                norm_output,
                actual_norm_output
            )
        ) {
            return EXIT_FAILURE;
        }

        if (
            !calculate_metrics(
                actual_norm_output,
                expected_norm_output,
                norm_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        /*
         * 在Final RMSNorm网络释放前取得独立Mat，
         * 并将其直接交给LM Head。
         */
        norm_handoff = norm_output.clone();

        if (norm_handoff.empty()) {
            std::cerr
                << "Final RMSNorm out0.clone()失败。\n";

            return EXIT_FAILURE;
        }
    }

    std::cout
        << "\n===== Final RMSNorm parity =====\n";

    print_metrics(
        "Final RMSNorm output",
        norm_metrics
    );

    const bool norm_passed =
        norm_metrics.maximum_abs_error
            <= 1.0e-5
        && norm_metrics.mean_abs_error
            <= 1.0e-6
        && norm_metrics.cosine_similarity
            >= 0.999999999;

    if (!norm_passed) {
        std::cerr
            << "❌ Final RMSNorm单Token误差超限。\n";

        return EXIT_FAILURE;
    }

    Metrics logits_metrics;
    std::vector<float> actual_logits;

    {
        ncnn::Net lm_head_network;

        lm_head_network.opt.use_vulkan_compute =
            false;

        lm_head_network.opt.use_packing_layout =
            use_packing_layout;

        lm_head_network.opt.num_threads =
            kThreads;

        const std::string param_path =
            lm_head_directory
            + "/lm_head.ncnn.param";

        const std::string model_path =
            lm_head_directory
            + "/lm_head.ncnn.bin";

        if (
            lm_head_network.load_param(
                param_path.c_str()
            ) != 0
        ) {
            std::cerr
                << "LM Head param加载失败："
                << param_path
                << '\n';

            return EXIT_FAILURE;
        }

        if (
            lm_head_network.load_model(
                model_path.c_str()
            ) != 0
        ) {
            std::cerr
                << "LM Head bin加载失败："
                << model_path
                << '\n';

            return EXIT_FAILURE;
        }

        ncnn::Extractor extractor =
            lm_head_network.create_extractor();

        extractor.set_light_mode(false);

        const int input_status =
            extractor.input(
                "in0",
                norm_handoff
            );

        if (input_status != 0) {
            std::cerr
                << "LM Head输入绑定失败："
                << input_status
                << '\n';

            return EXIT_FAILURE;
        }

        ncnn::Mat logits;

        const int output_status =
            extractor.extract(
                "out0",
                logits
            );

        if (output_status != 0) {
            std::cerr
                << "LM Head输出提取失败："
                << output_status
                << '\n';

            return EXIT_FAILURE;
        }

        if (logits.empty()) {
            std::cerr
                << "LM Head输出为空。\n";

            return EXIT_FAILURE;
        }

        std::cout
            << "\n===== LM Head output =====\n";

        print_shape(
            "decode_logits",
            logits
        );

        if (
            !unpack_mat(
                logits,
                actual_logits
            )
        ) {
            return EXIT_FAILURE;
        }

        if (
            actual_logits.size()
            != kDecodeVocabSize
        ) {
            std::cerr
                << "LM Head输出元素数错误："
                << actual_logits.size()
                << "，预期"
                << kDecodeVocabSize
                << '\n';

            return EXIT_FAILURE;
        }

        if (
            !calculate_metrics(
                actual_logits,
                expected_logits,
                logits_metrics
            )
        ) {
            return EXIT_FAILURE;
        }
    }

    const auto expected_iterator =
        std::max_element(
            expected_logits.begin(),
            expected_logits.end()
        );

    const auto actual_iterator =
        std::max_element(
            actual_logits.begin(),
            actual_logits.end()
        );

    const int expected_token =
        static_cast<int>(
            std::distance(
                expected_logits.begin(),
                expected_iterator
            )
        );

    const int actual_token =
        static_cast<int>(
            std::distance(
                actual_logits.begin(),
                actual_iterator
            )
        );

    std::cout
        << "\n===== Decode logits parity =====\n";

    print_metrics(
        "Final RMSNorm -> LM Head logits",
        logits_metrics
    );

    std::cout
        << "Expected decode token : "
        << expected_token
        << '\n'
        << "Actual decode token   : "
        << actual_token
        << '\n';

    /*
     * 使用现有独立LM Head测试已经采用的验收阈值。
     * 实际测得误差通常会显著低于这些上限。
     */
    const bool logits_passed =
        logits_metrics.maximum_abs_error
            <= 3.0e-3
        && logits_metrics.mean_abs_error
            <= 5.0e-5
        && logits_metrics.cosine_similarity
            >= 0.999999
        && actual_token == expected_token;

    if (!logits_passed) {
        std::cerr
            << "❌ Decode tail logits数值验证失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "\n✅ Layer23 reference output"
        << " → Final RMSNorm"
        << " → LM Head"
        << " → decode logits"
        << " ncnn直接串联验证成功。\n";

    return EXIT_SUCCESS;
}
