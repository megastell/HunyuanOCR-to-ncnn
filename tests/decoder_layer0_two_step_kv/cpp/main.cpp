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
constexpr int kMaskLength = 315;

constexpr int kRopeComponents = 4;
constexpr int kHeadDim = 128;

constexpr int kKeyValueHeads = 8;
constexpr int kPastLength = 314;
constexpr int kPresentLength = 315;

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



} // namespace


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

    constexpr int kPilotHiddenSize = 1024;
    constexpr int kPilotHeadDim = 128;
    constexpr int kPilotKvHeads = 8;
    constexpr int kPilotRopeComponents = 4;

    constexpr int kStep1MaskLength = 314;
    constexpr int kStep1PastLength = 313;
    constexpr int kStep1PresentLength = 314;

    constexpr int kStep2MaskLength = 315;
    constexpr int kStep2PastLength = 314;
    constexpr int kStep2PresentLength = 315;

    constexpr std::size_t kPilotHiddenCount =
        static_cast<std::size_t>(
            kPilotHiddenSize
        );

    constexpr std::size_t kPilotRopeCount =
        static_cast<std::size_t>(
            kPilotRopeComponents
        )
        * static_cast<std::size_t>(
            kPilotHeadDim
        );

    constexpr std::size_t kStep1MaskCount =
        static_cast<std::size_t>(
            kStep1MaskLength
        );

    constexpr std::size_t kStep2MaskCount =
        static_cast<std::size_t>(
            kStep2MaskLength
        );

    constexpr std::size_t kStep1PastCount =
        static_cast<std::size_t>(
            kPilotKvHeads
        )
        * static_cast<std::size_t>(
            kStep1PastLength
        )
        * static_cast<std::size_t>(
            kPilotHeadDim
        );

    constexpr std::size_t kStep1PresentCount =
        static_cast<std::size_t>(
            kPilotKvHeads
        )
        * static_cast<std::size_t>(
            kStep1PresentLength
        )
        * static_cast<std::size_t>(
            kPilotHeadDim
        );

    constexpr std::size_t kStep2PastCount =
        static_cast<std::size_t>(
            kPilotKvHeads
        )
        * static_cast<std::size_t>(
            kStep2PastLength
        )
        * static_cast<std::size_t>(
            kPilotHeadDim
        );

    constexpr std::size_t kStep2PresentCount =
        static_cast<std::size_t>(
            kPilotKvHeads
        )
        * static_cast<std::size_t>(
            kStep2PresentLength
        )
        * static_cast<std::size_t>(
            kPilotHeadDim
        );

    static_assert(
        kStep1PresentCount
        == kStep2PastCount
    );

    const std::string step1_directory =
        project_root
        + "/artifacts/decoder_layer0_decode";

    const std::string step2_directory =
        project_root
        + "/artifacts/decoder_layer0_decode_step2";

    const std::string step1_reference =
        step1_directory
        + "/reference";

    const std::string step2_reference =
        step2_directory
        + "/reference";

    std::vector<float> step1_hidden_values;
    std::vector<float> step1_mask_values;
    std::vector<float> step1_rope_cos_values;
    std::vector<float> step1_rope_sin_values;
    std::vector<float> step1_past_key_values;
    std::vector<float> step1_past_value_values;

    std::vector<float> expected_step1_output;
    std::vector<float> expected_step1_present_key;
    std::vector<float> expected_step1_present_value;

    std::vector<float> step2_hidden_values;
    std::vector<float> step2_mask_values;
    std::vector<float> step2_rope_cos_values;
    std::vector<float> step2_rope_sin_values;

    /*
     * Step 2参考past KV只用于边界数值检查，
     * 不会被绑定到Step 2 ncnn网络。
     */
    std::vector<float> expected_step2_past_key;
    std::vector<float> expected_step2_past_value;

    std::vector<float> expected_step2_output;
    std::vector<float> expected_step2_present_key;
    std::vector<float> expected_step2_present_value;

    if (
        !load_exact_binary(
            step1_reference
                + "/layer0_hidden_states_f32.bin",
            kPilotHiddenCount,
            step1_hidden_values
        )
        || !load_exact_binary(
            step1_reference
                + "/layer0_attention_mask_f32.bin",
            kStep1MaskCount,
            step1_mask_values
        )
        || !load_exact_binary(
            step1_reference
                + "/layer0_position_embeddings_0_f32.bin",
            kPilotRopeCount,
            step1_rope_cos_values
        )
        || !load_exact_binary(
            step1_reference
                + "/layer0_position_embeddings_1_f32.bin",
            kPilotRopeCount,
            step1_rope_sin_values
        )
        || !load_exact_binary(
            step1_reference
                + "/past_key_f32.bin",
            kStep1PastCount,
            step1_past_key_values
        )
        || !load_exact_binary(
            step1_reference
                + "/past_value_f32.bin",
            kStep1PastCount,
            step1_past_value_values
        )
        || !load_exact_binary(
            step1_reference
                + "/layer0_output_f32.bin",
            kPilotHiddenCount,
            expected_step1_output
        )
        || !load_exact_binary(
            step1_reference
                + "/present_key_f32.bin",
            kStep1PresentCount,
            expected_step1_present_key
        )
        || !load_exact_binary(
            step1_reference
                + "/present_value_f32.bin",
            kStep1PresentCount,
            expected_step1_present_value
        )
    ) {
        std::cerr
            << "Step 1参考数据加载失败。\n";

        return EXIT_FAILURE;
    }

    if (
        !load_exact_binary(
            step2_reference
                + "/layer0_hidden_states_f32.bin",
            kPilotHiddenCount,
            step2_hidden_values
        )
        || !load_exact_binary(
            step2_reference
                + "/layer0_attention_mask_f32.bin",
            kStep2MaskCount,
            step2_mask_values
        )
        || !load_exact_binary(
            step2_reference
                + "/layer0_position_embeddings_0_f32.bin",
            kPilotRopeCount,
            step2_rope_cos_values
        )
        || !load_exact_binary(
            step2_reference
                + "/layer0_position_embeddings_1_f32.bin",
            kPilotRopeCount,
            step2_rope_sin_values
        )
        || !load_exact_binary(
            step2_reference
                + "/past_key_f32.bin",
            kStep2PastCount,
            expected_step2_past_key
        )
        || !load_exact_binary(
            step2_reference
                + "/past_value_f32.bin",
            kStep2PastCount,
            expected_step2_past_value
        )
        || !load_exact_binary(
            step2_reference
                + "/layer0_output_f32.bin",
            kPilotHiddenCount,
            expected_step2_output
        )
        || !load_exact_binary(
            step2_reference
                + "/present_key_f32.bin",
            kStep2PresentCount,
            expected_step2_present_key
        )
        || !load_exact_binary(
            step2_reference
                + "/present_value_f32.bin",
            kStep2PresentCount,
            expected_step2_present_value
        )
    ) {
        std::cerr
            << "Step 2参考数据加载失败。\n";

        return EXIT_FAILURE;
    }

    auto make_hidden = [](
        std::vector<float>& values)
    {
        ncnn::Mat flat(
            static_cast<int>(
                values.size()
            ),
            static_cast<void*>(
                values.data()
            )
        );

        return flat.reshape(
            kPilotHiddenSize,
            1
        ).clone();
    };

    auto make_mask = [](
        std::vector<float>& values,
        int length)
    {
        ncnn::Mat flat(
            static_cast<int>(
                values.size()
            ),
            static_cast<void*>(
                values.data()
            )
        );

        return flat.reshape(
            length,
            1,
            1
        ).clone();
    };

    auto make_rope = [](
        std::vector<float>& values)
    {
        ncnn::Mat flat(
            static_cast<int>(
                values.size()
            ),
            static_cast<void*>(
                values.data()
            )
        );

        return flat.reshape(
            kPilotHeadDim,
            1,
            kPilotRopeComponents
        ).clone();
    };

    auto make_cache = [](
        std::vector<float>& values,
        int sequence_length)
    {
        ncnn::Mat flat(
            static_cast<int>(
                values.size()
            ),
            static_cast<void*>(
                values.data()
            )
        );

        return flat.reshape(
            kPilotHeadDim,
            sequence_length,
            kPilotKvHeads
        ).clone();
    };

    auto cache_region_max_error = [](
        const std::vector<float>& actual,
        int actual_length,
        const std::vector<float>& expected,
        int expected_length,
        int begin_position,
        int end_position)
    {
        double maximum_error = 0.0;

        for (
            int head = 0;
            head < kPilotKvHeads;
            ++head
        ) {
            for (
                int position = begin_position;
                position < end_position;
                ++position
            ) {
                for (
                    int dimension = 0;
                    dimension < kPilotHeadDim;
                    ++dimension
                ) {
                    const std::size_t actual_index =
                        (
                            (
                                static_cast<std::size_t>(
                                    head
                                )
                                * static_cast<std::size_t>(
                                    actual_length
                                )
                                + static_cast<std::size_t>(
                                    position
                                )
                            )
                            * static_cast<std::size_t>(
                                kPilotHeadDim
                            )
                        )
                        + static_cast<std::size_t>(
                            dimension
                        );

                    const std::size_t expected_index =
                        (
                            (
                                static_cast<std::size_t>(
                                    head
                                )
                                * static_cast<std::size_t>(
                                    expected_length
                                )
                                + static_cast<std::size_t>(
                                    position
                                )
                            )
                            * static_cast<std::size_t>(
                                kPilotHeadDim
                            )
                        )
                        + static_cast<std::size_t>(
                            dimension
                        );

                    const double error =
                        std::fabs(
                            static_cast<double>(
                                actual[actual_index]
                            )
                            - static_cast<double>(
                                expected[expected_index]
                            )
                        );

                    maximum_error =
                        std::max(
                            maximum_error,
                            error
                        );
                }
            }
        }

        return maximum_error;
    };

    auto metrics_passed = [](
        const Metrics& metrics,
        double maximum_tolerance,
        double mean_tolerance)
    {
        return
            metrics.maximum_abs_error
                <= maximum_tolerance
            && metrics.mean_abs_error
                <= mean_tolerance
            && metrics.cosine_similarity
                >= 0.99999999;
    };

    ncnn::Mat step1_hidden =
        make_hidden(
            step1_hidden_values
        );

    ncnn::Mat step1_mask =
        make_mask(
            step1_mask_values,
            kStep1MaskLength
        );

    ncnn::Mat step1_rope_cos =
        make_rope(
            step1_rope_cos_values
        );

    ncnn::Mat step1_rope_sin =
        make_rope(
            step1_rope_sin_values
        );

    ncnn::Mat step1_past_key =
        make_cache(
            step1_past_key_values,
            kStep1PastLength
        );

    ncnn::Mat step1_past_value =
        make_cache(
            step1_past_value_values,
            kStep1PastLength
        );

    ncnn::Mat step2_hidden =
        make_hidden(
            step2_hidden_values
        );

    ncnn::Mat step2_mask =
        make_mask(
            step2_mask_values,
            kStep2MaskLength
        );

    ncnn::Mat step2_rope_cos =
        make_rope(
            step2_rope_cos_values
        );

    ncnn::Mat step2_rope_sin =
        make_rope(
            step2_rope_sin_values
        );

    if (
        step1_hidden.empty()
        || step1_mask.empty()
        || step1_rope_cos.empty()
        || step1_rope_sin.empty()
        || step1_past_key.empty()
        || step1_past_value.empty()
        || step2_hidden.empty()
        || step2_mask.empty()
        || step2_rope_cos.empty()
        || step2_rope_sin.empty()
    ) {
        std::cerr
            << "输入ncnn::Mat创建失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "===== Two-step Layer 0 KV runtime =====\n"
        << "threads                  : "
        << kThreads
        << '\n'
        << "packing layout           : "
        << (
            use_packing_layout
            ? "true"
            : "false"
        )
        << '\n'
        << "Step 1 past length       : "
        << kStep1PastLength
        << '\n'
        << "Step 1 present length    : "
        << kStep1PresentLength
        << '\n'
        << "Step 2 past length       : "
        << kStep2PastLength
        << '\n'
        << "Step 2 present length    : "
        << kStep2PresentLength
        << '\n'
        << "KV file reload at Step 2: disabled\n"
        << "KV handoff               : "
        << "Step1 out1/out2.clone()"
        << " -> Step2 in4/in5\n";

    Metrics step1_output_metrics;
    Metrics step1_key_metrics;
    Metrics step1_value_metrics;

    ncnn::Mat key_handoff;
    ncnn::Mat value_handoff;

    {
        ncnn::Net network;

        network.opt.use_vulkan_compute =
            false;

        network.opt.use_packing_layout =
            use_packing_layout;

        network.opt.num_threads =
            kThreads;

        const std::string param_path =
            step1_directory
            + "/decoder_layer0_decode.ncnn.param";

        const std::string model_path =
            step1_directory
            + "/decoder_layer0_decode.ncnn.bin";

        if (
            network.load_param(
                param_path.c_str()
            ) != 0
            || network.load_model(
                model_path.c_str()
            ) != 0
        ) {
            std::cerr
                << "Step 1 ncnn模型加载失败。\n";

            return EXIT_FAILURE;
        }

        ncnn::Extractor extractor =
            network.create_extractor();

        extractor.set_light_mode(false);

        if (
            extractor.input(
                "in0",
                step1_hidden
            ) != 0
            || extractor.input(
                "in1",
                step1_mask
            ) != 0
            || extractor.input(
                "in2",
                step1_rope_cos
            ) != 0
            || extractor.input(
                "in3",
                step1_rope_sin
            ) != 0
            || extractor.input(
                "in4",
                step1_past_key
            ) != 0
            || extractor.input(
                "in5",
                step1_past_value
            ) != 0
        ) {
            std::cerr
                << "Step 1输入绑定失败。\n";

            return EXIT_FAILURE;
        }

        ncnn::Mat output;
        ncnn::Mat present_key;
        ncnn::Mat present_value;

        if (
            extractor.extract(
                "out0",
                output
            ) != 0
            || extractor.extract(
                "out1",
                present_key
            ) != 0
            || extractor.extract(
                "out2",
                present_value
            ) != 0
        ) {
            std::cerr
                << "Step 1输出提取失败。\n";

            return EXIT_FAILURE;
        }

        std::cout
            << "\n===== Step 1 output shapes =====\n";

        print_shape(
            "step1_out0",
            output
        );

        print_shape(
            "step1_out1_present_key",
            present_key
        );

        print_shape(
            "step1_out2_present_value",
            present_value
        );

        std::vector<float> actual_output;
        std::vector<float> actual_key;
        std::vector<float> actual_value;

        if (
            !unpack_mat(
                output,
                actual_output
            )
            || !unpack_mat(
                present_key,
                actual_key
            )
            || !unpack_mat(
                present_value,
                actual_value
            )
        ) {
            return EXIT_FAILURE;
        }

        if (
            !calculate_metrics(
                actual_output,
                expected_step1_output,
                step1_output_metrics
            )
            || !calculate_metrics(
                actual_key,
                expected_step1_present_key,
                step1_key_metrics
            )
            || !calculate_metrics(
                actual_value,
                expected_step1_present_value,
                step1_value_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        /*
         * 必须在Step 1 network释放前复制KV。
         * 这两个独立Mat将成为Step 2的真实past KV。
         */
        key_handoff =
            present_key.clone();

        value_handoff =
            present_value.clone();

        if (
            key_handoff.empty()
            || value_handoff.empty()
        ) {
            std::cerr
                << "Step 1 present KV clone失败。\n";

            return EXIT_FAILURE;
        }
    }

    std::cout
        << "\n===== Step 1 numerical parity =====\n";

    print_metrics(
        "Step 1 layer output",
        step1_output_metrics
    );

    print_metrics(
        "Step 1 present key",
        step1_key_metrics
    );

    print_metrics(
        "Step 1 present value",
        step1_value_metrics
    );

    std::vector<float> handoff_key_values;
    std::vector<float> handoff_value_values;

    if (
        !unpack_mat(
            key_handoff,
            handoff_key_values
        )
        || !unpack_mat(
            value_handoff,
            handoff_value_values
        )
    ) {
        return EXIT_FAILURE;
    }

    Metrics key_boundary_metrics;
    Metrics value_boundary_metrics;

    if (
        !calculate_metrics(
            handoff_key_values,
            expected_step2_past_key,
            key_boundary_metrics
        )
        || !calculate_metrics(
            handoff_value_values,
            expected_step2_past_value,
            value_boundary_metrics
        )
    ) {
        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Step 1 -> Step 2 KV boundary =====\n";

    print_shape(
        "step2_in4_direct_key",
        key_handoff
    );

    print_shape(
        "step2_in5_direct_value",
        value_handoff
    );

    print_metrics(
        "Step 1 ncnn present key vs Step 2 PyTorch past key",
        key_boundary_metrics
    );

    print_metrics(
        "Step 1 ncnn present value vs Step 2 PyTorch past value",
        value_boundary_metrics
    );

    Metrics step2_output_metrics;
    Metrics step2_key_metrics;
    Metrics step2_value_metrics;

    double key_handoff_prefix_error = 0.0;
    double value_handoff_prefix_error = 0.0;

    double step2_appended_key_error = 0.0;
    double step2_appended_value_error = 0.0;

    {
        ncnn::Net network;

        network.opt.use_vulkan_compute =
            false;

        network.opt.use_packing_layout =
            use_packing_layout;

        network.opt.num_threads =
            kThreads;

        const std::string param_path =
            step2_directory
            + "/decoder_layer0_decode_step2.ncnn.param";

        const std::string model_path =
            step2_directory
            + "/decoder_layer0_decode_step2.ncnn.bin";

        if (
            network.load_param(
                param_path.c_str()
            ) != 0
            || network.load_model(
                model_path.c_str()
            ) != 0
        ) {
            std::cerr
                << "Step 2 ncnn模型加载失败。\n";

            return EXIT_FAILURE;
        }

        ncnn::Extractor extractor =
            network.create_extractor();

        extractor.set_light_mode(false);

        /*
         * in0、mask和RoPE来自Step 2参考契约。
         * in4和in5只能来自Step 1 ncnn输出。
         */
        if (
            extractor.input(
                "in0",
                step2_hidden
            ) != 0
            || extractor.input(
                "in1",
                step2_mask
            ) != 0
            || extractor.input(
                "in2",
                step2_rope_cos
            ) != 0
            || extractor.input(
                "in3",
                step2_rope_sin
            ) != 0
            || extractor.input(
                "in4",
                key_handoff
            ) != 0
            || extractor.input(
                "in5",
                value_handoff
            ) != 0
        ) {
            std::cerr
                << "Step 2输入绑定失败。\n";

            return EXIT_FAILURE;
        }

        ncnn::Mat output;
        ncnn::Mat present_key;
        ncnn::Mat present_value;

        if (
            extractor.extract(
                "out0",
                output
            ) != 0
            || extractor.extract(
                "out1",
                present_key
            ) != 0
            || extractor.extract(
                "out2",
                present_value
            ) != 0
        ) {
            std::cerr
                << "Step 2输出提取失败。\n";

            return EXIT_FAILURE;
        }

        std::cout
            << "\n===== Step 2 output shapes =====\n";

        print_shape(
            "step2_out0",
            output
        );

        print_shape(
            "step2_out1_present_key",
            present_key
        );

        print_shape(
            "step2_out2_present_value",
            present_value
        );

        std::vector<float> actual_output;
        std::vector<float> actual_key;
        std::vector<float> actual_value;

        if (
            !unpack_mat(
                output,
                actual_output
            )
            || !unpack_mat(
                present_key,
                actual_key
            )
            || !unpack_mat(
                present_value,
                actual_value
            )
        ) {
            return EXIT_FAILURE;
        }

        if (
            !calculate_metrics(
                actual_output,
                expected_step2_output,
                step2_output_metrics
            )
            || !calculate_metrics(
                actual_key,
                expected_step2_present_key,
                step2_key_metrics
            )
            || !calculate_metrics(
                actual_value,
                expected_step2_present_value,
                step2_value_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        /*
         * Step 2输出缓存前314个位置必须逐元素等于
         * 传入的Step 1 ncnn present KV。
         */
        key_handoff_prefix_error =
            cache_region_max_error(
                actual_key,
                kStep2PresentLength,
                handoff_key_values,
                kStep2PastLength,
                0,
                kStep2PastLength
            );

        value_handoff_prefix_error =
            cache_region_max_error(
                actual_value,
                kStep2PresentLength,
                handoff_value_values,
                kStep2PastLength,
                0,
                kStep2PastLength
            );

        step2_appended_key_error =
            cache_region_max_error(
                actual_key,
                kStep2PresentLength,
                expected_step2_present_key,
                kStep2PresentLength,
                kStep2PastLength,
                kStep2PresentLength
            );

        step2_appended_value_error =
            cache_region_max_error(
                actual_value,
                kStep2PresentLength,
                expected_step2_present_value,
                kStep2PresentLength,
                kStep2PastLength,
                kStep2PresentLength
            );
    }

    std::cout
        << "\n===== Step 2 direct-KV numerical parity =====\n";

    print_metrics(
        "Step 2 layer output",
        step2_output_metrics
    );

    print_metrics(
        "Step 2 present key",
        step2_key_metrics
    );

    print_metrics(
        "Step 2 present value",
        step2_value_metrics
    );

    std::cout
        << std::scientific
        << std::setprecision(10)
        << "\n===== Direct KV cache checks =====\n"
        << "Step 2 key prefix vs Step 1 handoff   : "
        << key_handoff_prefix_error
        << '\n'
        << "Step 2 value prefix vs Step 1 handoff : "
        << value_handoff_prefix_error
        << '\n'
        << "Step 2 appended key vs reference      : "
        << step2_appended_key_error
        << '\n'
        << "Step 2 appended value vs reference    : "
        << step2_appended_value_error
        << '\n';

    const bool step1_passed =
        metrics_passed(
            step1_output_metrics,
            1.0e-5,
            1.0e-7
        )
        && metrics_passed(
            step1_key_metrics,
            5.0e-6,
            1.0e-7
        )
        && metrics_passed(
            step1_value_metrics,
            1.0e-6,
            1.0e-8
        );

    const bool boundary_passed =
        metrics_passed(
            key_boundary_metrics,
            5.0e-6,
            1.0e-7
        )
        && metrics_passed(
            value_boundary_metrics,
            1.0e-6,
            1.0e-8
        );

    const bool step2_passed =
        metrics_passed(
            step2_output_metrics,
            2.0e-5,
            1.0e-6
        )
        && metrics_passed(
            step2_key_metrics,
            1.0e-5,
            1.0e-7
        )
        && metrics_passed(
            step2_value_metrics,
            2.0e-6,
            1.0e-8
        )
        && key_handoff_prefix_error
            == 0.0
        && value_handoff_prefix_error
            == 0.0
        && step2_appended_key_error
            <= 5.0e-6
        && step2_appended_value_error
            <= 1.0e-6;

    if (
        !step1_passed
        || !boundary_passed
        || !step2_passed
    ) {
        std::cerr
            << "\n❌ Decoder Layer 0两步KV直连"
            << "数值验证失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "\n✅ Decoder Layer 0 Step 1"
        << " present KV"
        << " → Step 2 past KV"
        << " ncnn::Mat直接串联验证成功。\n";

    return EXIT_SUCCESS;
}
