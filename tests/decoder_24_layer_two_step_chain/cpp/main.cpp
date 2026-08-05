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

    constexpr int kLayerCount = 24;
    constexpr int kRuntimeThreads = 9;

    constexpr int kHiddenSize = 1024;
    constexpr int kKvHeads = 8;
    constexpr int kHeadDim = 128;
    constexpr int kRopeComponents = 4;

    constexpr int kStep1MaskLength = 314;
    constexpr int kStep1PastLength = 313;
    constexpr int kStep1PresentLength = 314;

    constexpr int kStep2MaskLength = 315;
    constexpr int kStep2PastLength = 314;
    constexpr int kStep2PresentLength = 315;

    const std::size_t hidden_count =
        static_cast<std::size_t>(
            kHiddenSize
        );

    const std::size_t rope_count =
        static_cast<std::size_t>(
            kRopeComponents
        )
        * static_cast<std::size_t>(
            kHeadDim
        );

    const std::size_t step1_mask_count =
        static_cast<std::size_t>(
            kStep1MaskLength
        );

    const std::size_t step2_mask_count =
        static_cast<std::size_t>(
            kStep2MaskLength
        );

    const std::size_t step1_past_count =
        static_cast<std::size_t>(
            kKvHeads
        )
        * static_cast<std::size_t>(
            kStep1PastLength
        )
        * static_cast<std::size_t>(
            kHeadDim
        );

    const std::size_t step1_present_count =
        static_cast<std::size_t>(
            kKvHeads
        )
        * static_cast<std::size_t>(
            kStep1PresentLength
        )
        * static_cast<std::size_t>(
            kHeadDim
        );

    const std::size_t step2_past_count =
        static_cast<std::size_t>(
            kKvHeads
        )
        * static_cast<std::size_t>(
            kStep2PastLength
        )
        * static_cast<std::size_t>(
            kHeadDim
        );

    const std::size_t step2_present_count =
        static_cast<std::size_t>(
            kKvHeads
        )
        * static_cast<std::size_t>(
            kStep2PresentLength
        )
        * static_cast<std::size_t>(
            kHeadDim
        );

    if (
        step1_present_count
        != step2_past_count
    ) {
        std::cerr
            << "Step 1 present与Step 2 past"
            << "元素数量不一致。\n";

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
            kHiddenSize,
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
            kHeadDim,
            1,
            kRopeComponents
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
            kHeadDim,
            sequence_length,
            kKvHeads
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
            head < kKvHeads;
            ++head
        ) {
            for (
                int position = begin_position;
                position < end_position;
                ++position
            ) {
                for (
                    int dimension = 0;
                    dimension < kHeadDim;
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
                                kHeadDim
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
                                kHeadDim
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

    auto hidden_passed = [](
        const Metrics& metrics)
    {
        return
            metrics.maximum_abs_error
                <= 2.0e-3
            && metrics.mean_abs_error
                <= 2.0e-4
            && metrics.cosine_similarity
                >= 0.999999;
    };

    auto cache_passed = [](
        const Metrics& metrics)
    {
        return
            metrics.maximum_abs_error
                <= 2.0e-4
            && metrics.mean_abs_error
                <= 2.0e-6
            && metrics.cosine_similarity
                >= 0.999999;
    };

    const std::string step1_layer0_reference =
        project_root
        + "/artifacts/decoder_layer0_decode/reference";

    const std::string step2_layer0_reference =
        project_root
        + "/artifacts/decoder_layer0_decode_step2/reference";

    std::vector<float> step1_initial_hidden_values;
    std::vector<float> step1_mask_values;
    std::vector<float> step1_rope_cos_values;
    std::vector<float> step1_rope_sin_values;

    std::vector<float> step2_mask_values;
    std::vector<float> step2_rope_cos_values;
    std::vector<float> step2_rope_sin_values;

    if (
        !load_exact_binary(
            step1_layer0_reference
                + "/layer0_hidden_states_f32.bin",
            hidden_count,
            step1_initial_hidden_values
        )
        || !load_exact_binary(
            step1_layer0_reference
                + "/layer0_attention_mask_f32.bin",
            step1_mask_count,
            step1_mask_values
        )
        || !load_exact_binary(
            step1_layer0_reference
                + "/layer0_position_embeddings_0_f32.bin",
            rope_count,
            step1_rope_cos_values
        )
        || !load_exact_binary(
            step1_layer0_reference
                + "/layer0_position_embeddings_1_f32.bin",
            rope_count,
            step1_rope_sin_values
        )
        || !load_exact_binary(
            step2_layer0_reference
                + "/layer0_attention_mask_f32.bin",
            step2_mask_count,
            step2_mask_values
        )
        || !load_exact_binary(
            step2_layer0_reference
                + "/layer0_position_embeddings_0_f32.bin",
            rope_count,
            step2_rope_cos_values
        )
        || !load_exact_binary(
            step2_layer0_reference
                + "/layer0_position_embeddings_1_f32.bin",
            rope_count,
            step2_rope_sin_values
        )
    ) {
        std::cerr
            << "共享输入张量加载失败。\n";

        return EXIT_FAILURE;
    }

    ncnn::Mat step1_current_hidden =
        make_hidden(
            step1_initial_hidden_values
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
        step1_current_hidden.empty()
        || step1_mask.empty()
        || step1_rope_cos.empty()
        || step1_rope_sin.empty()
        || step2_mask.empty()
        || step2_rope_cos.empty()
        || step2_rope_sin.empty()
    ) {
        std::cerr
            << "共享ncnn::Mat创建失败。\n";

        return EXIT_FAILURE;
    }

    std::vector<ncnn::Mat>
        step1_present_keys(
            kLayerCount
        );

    std::vector<ncnn::Mat>
        step1_present_values(
            kLayerCount
        );

    double step1_maximum_boundary_error = 0.0;
    double step1_maximum_output_error = 0.0;
    double step1_maximum_key_error = 0.0;
    double step1_maximum_value_error = 0.0;

    double step2_maximum_boundary_error = 0.0;
    double step2_maximum_output_error = 0.0;
    double step2_maximum_key_error = 0.0;
    double step2_maximum_value_error = 0.0;

    double maximum_handoff_key_error = 0.0;
    double maximum_handoff_value_error = 0.0;

    double maximum_step2_key_prefix_error = 0.0;
    double maximum_step2_value_prefix_error = 0.0;

    double maximum_step2_appended_key_error = 0.0;
    double maximum_step2_appended_value_error = 0.0;

    std::cout
        << "===== Decoder 24-layer two-step runtime =====\n"
        << "threads                    : "
        << kRuntimeThreads
        << '\n'
        << "packing layout             : "
        << (
            use_packing_layout
            ? "true"
            : "false"
        )
        << '\n'
        << "Step 1 past/present        : "
        << kStep1PastLength
        << " -> "
        << kStep1PresentLength
        << '\n'
        << "Step 2 past/present        : "
        << kStep2PastLength
        << " -> "
        << kStep2PresentLength
        << '\n'
        << "Step 1 hidden reload       : disabled\n"
        << "Step 2 hidden reload       : disabled\n"
        << "Step 2 initial hidden      : Step1 token -> ncnn Embed\n"
        << "Step 2 KV inference reload : disabled\n"
        << "hidden handoff             : "
        << "previous out0.clone() -> next in0\n"
        << "KV handoff                 : "
        << "Step1 Layer i out1/out2.clone()"
        << " -> Step2 Layer i in4/in5\n";

    std::cout
        << "\n===== Step 1: 24-layer hidden chain =====\n";

    for (
        int layer = 0;
        layer < kLayerCount;
        ++layer
    ) {
        const std::string layer_text =
            std::to_string(layer);

        const std::string name =
            "decoder_layer"
            + layer_text
            + "_decode";

        const std::string model_directory =
            project_root
            + "/artifacts/"
            + name;

        const std::string reference =
            model_directory
            + "/reference";

        std::vector<float> expected_hidden;
        std::vector<float> past_key_values;
        std::vector<float> past_value_values;
        std::vector<float> expected_output;
        std::vector<float> expected_present_key;
        std::vector<float> expected_present_value;

        if (
            !load_exact_binary(
                reference
                    + "/layer"
                    + layer_text
                    + "_hidden_states_f32.bin",
                hidden_count,
                expected_hidden
            )
            || !load_exact_binary(
                reference
                    + "/past_key_f32.bin",
                step1_past_count,
                past_key_values
            )
            || !load_exact_binary(
                reference
                    + "/past_value_f32.bin",
                step1_past_count,
                past_value_values
            )
            || !load_exact_binary(
                reference
                    + "/layer"
                    + layer_text
                    + "_output_f32.bin",
                hidden_count,
                expected_output
            )
            || !load_exact_binary(
                reference
                    + "/present_key_f32.bin",
                step1_present_count,
                expected_present_key
            )
            || !load_exact_binary(
                reference
                    + "/present_value_f32.bin",
                step1_present_count,
                expected_present_value
            )
        ) {
            std::cerr
                << "Step 1 Layer "
                << layer
                << "参考数据加载失败。\n";

            return EXIT_FAILURE;
        }

        std::vector<float> actual_input;

        if (
            !unpack_mat(
                step1_current_hidden,
                actual_input
            )
        ) {
            return EXIT_FAILURE;
        }

        Metrics input_metrics;

        if (
            !calculate_metrics(
                actual_input,
                expected_hidden,
                input_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        ncnn::Mat past_key =
            make_cache(
                past_key_values,
                kStep1PastLength
            );

        ncnn::Mat past_value =
            make_cache(
                past_value_values,
                kStep1PastLength
            );

        if (
            past_key.empty()
            || past_value.empty()
        ) {
            std::cerr
                << "Step 1 Layer "
                << layer
                << " past KV Mat创建失败。\n";

            return EXIT_FAILURE;
        }

        ncnn::Mat next_hidden;
        ncnn::Mat present_key;
        ncnn::Mat present_value;

        {
            ncnn::Net network;

            network.opt.use_vulkan_compute =
                false;

            network.opt.use_packing_layout =
                use_packing_layout;

            network.opt.num_threads =
                kRuntimeThreads;

            const std::string param_path =
                model_directory
                + "/"
                + name
                + ".ncnn.param";

            const std::string model_path =
                model_directory
                + "/"
                + name
                + ".ncnn.bin";

            if (
                network.load_param(
                    param_path.c_str()
                ) != 0
                || network.load_model(
                    model_path.c_str()
                ) != 0
            ) {
                std::cerr
                    << "Step 1 Layer "
                    << layer
                    << "模型加载失败。\n";

                return EXIT_FAILURE;
            }

            ncnn::Extractor extractor =
                network.create_extractor();

            extractor.set_light_mode(false);

            if (
                extractor.input(
                    "in0",
                    step1_current_hidden
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
                    past_key
                ) != 0
                || extractor.input(
                    "in5",
                    past_value
                ) != 0
            ) {
                std::cerr
                    << "Step 1 Layer "
                    << layer
                    << "输入绑定失败。\n";

                return EXIT_FAILURE;
            }

            ncnn::Mat output;

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
                    << "Step 1 Layer "
                    << layer
                    << "输出提取失败。\n";

                return EXIT_FAILURE;
            }

            next_hidden =
                output.clone();

            step1_present_keys[layer] =
                present_key.clone();

            step1_present_values[layer] =
                present_value.clone();
        }

        if (
            next_hidden.empty()
            || step1_present_keys[layer].empty()
            || step1_present_values[layer].empty()
        ) {
            std::cerr
                << "Step 1 Layer "
                << layer
                << "输出clone失败。\n";

            return EXIT_FAILURE;
        }

        std::vector<float> actual_output;
        std::vector<float> actual_key;
        std::vector<float> actual_value;

        if (
            !unpack_mat(
                next_hidden,
                actual_output
            )
            || !unpack_mat(
                step1_present_keys[layer],
                actual_key
            )
            || !unpack_mat(
                step1_present_values[layer],
                actual_value
            )
        ) {
            return EXIT_FAILURE;
        }

        Metrics output_metrics;
        Metrics key_metrics;
        Metrics value_metrics;

        if (
            !calculate_metrics(
                actual_output,
                expected_output,
                output_metrics
            )
            || !calculate_metrics(
                actual_key,
                expected_present_key,
                key_metrics
            )
            || !calculate_metrics(
                actual_value,
                expected_present_value,
                value_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        const double key_prefix_error =
            cache_region_max_error(
                actual_key,
                kStep1PresentLength,
                expected_present_key,
                kStep1PresentLength,
                0,
                kStep1PastLength
            );

        const double value_prefix_error =
            cache_region_max_error(
                actual_value,
                kStep1PresentLength,
                expected_present_value,
                kStep1PresentLength,
                0,
                kStep1PastLength
            );

        step1_maximum_boundary_error =
            std::max(
                step1_maximum_boundary_error,
                input_metrics.maximum_abs_error
            );

        step1_maximum_output_error =
            std::max(
                step1_maximum_output_error,
                output_metrics.maximum_abs_error
            );

        step1_maximum_key_error =
            std::max(
                step1_maximum_key_error,
                key_metrics.maximum_abs_error
            );

        step1_maximum_value_error =
            std::max(
                step1_maximum_value_error,
                value_metrics.maximum_abs_error
            );

        std::cout
            << "Step1 Layer "
            << std::setw(2)
            << std::setfill('0')
            << layer
            << std::setfill(' ')
            << ": boundary="
            << std::scientific
            << std::setprecision(10)
            << input_metrics.maximum_abs_error
            << ", output="
            << output_metrics.maximum_abs_error
            << ", key="
            << key_metrics.maximum_abs_error
            << ", value="
            << value_metrics.maximum_abs_error
            << ", prefix=("
            << key_prefix_error
            << ", "
            << value_prefix_error
            << ")\n";

        const bool passed =
            hidden_passed(input_metrics)
            && hidden_passed(output_metrics)
            && cache_passed(key_metrics)
            && cache_passed(value_metrics)
            && key_prefix_error == 0.0
            && value_prefix_error == 0.0;

        if (!passed) {
            std::cerr
                << "❌ Step 1 Layer "
                << layer
                << "数值验证失败。\n";

            print_metrics(
                "input boundary",
                input_metrics
            );

            print_metrics(
                "layer output",
                output_metrics
            );

            print_metrics(
                "present key",
                key_metrics
            );

            print_metrics(
                "present value",
                value_metrics
            );

            return EXIT_FAILURE;
        }

        step1_current_hidden =
            next_hidden;
    }

    /*
     * 真正的自回归Token反馈：
     *
     * Step 1 Layer 23 out0
     *     -> Final RMSNorm
     *     -> LM Head
     *     -> argmax token 5112
     *     -> ncnn Embed
     *     -> Step 2 Layer 0 in0
     *
     * 参考hidden仅用于数值验证，不作为推理输入。
     */
    constexpr std::size_t kFeedbackVocabSize =
        120818;

    constexpr int kExpectedStep1Token =
        5112;

    int step1_actual_token =
        -1;

    ncnn::Mat step2_current_hidden;

    {
        const std::string step1_tail_reference =
            project_root
            + "/artifacts/decode_tail/reference";

        const std::string feedback_reference =
            project_root
            + "/artifacts/token_embedding_feedback/reference";

        const std::string final_norm_directory =
            project_root
            + "/artifacts/final_norm";

        const std::string lm_head_directory =
            project_root
            + "/artifacts/lm_head";

        const std::string embedding_directory =
            project_root
            + "/artifacts/text_embedding";

        std::vector<float>
            expected_step1_norm_input;

        std::vector<float>
            expected_step1_norm_output;

        std::vector<float>
            expected_step1_logits;

        std::vector<float>
            expected_step2_initial_hidden;

        if (
            !load_exact_binary(
                step1_tail_reference
                    + "/final_norm_input_f32.bin",
                hidden_count,
                expected_step1_norm_input
            )
            || !load_exact_binary(
                step1_tail_reference
                    + "/final_norm_output_f32.bin",
                hidden_count,
                expected_step1_norm_output
            )
            || !load_exact_binary(
                step1_tail_reference
                    + "/decode_logits_f32.bin",
                kFeedbackVocabSize,
                expected_step1_logits
            )
            || !load_exact_binary(
                feedback_reference
                    + "/token_5112_embedding_f32.bin",
                hidden_count,
                expected_step2_initial_hidden
            )
        ) {
            std::cerr
                << "Step 1反馈链参考数据加载失败。\n";

            return EXIT_FAILURE;
        }

        /*
         * Step 1 Decoder Layer 23
         * -> Final RMSNorm边界验证。
         */
        std::vector<float>
            actual_step1_decoder_output;

        if (
            !unpack_mat(
                step1_current_hidden,
                actual_step1_decoder_output
            )
        ) {
            return EXIT_FAILURE;
        }

        Metrics step1_norm_input_metrics;

        if (
            !calculate_metrics(
                actual_step1_decoder_output,
                expected_step1_norm_input,
                step1_norm_input_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        std::cout
            << "\n===== Step 1 Decoder -> Final RMSNorm boundary =====\n";

        print_metrics(
            "Step 1 Layer 23 chained output vs Final RMSNorm reference input",
            step1_norm_input_metrics
        );

        if (
            !hidden_passed(
                step1_norm_input_metrics
            )
        ) {
            std::cerr
                << "❌ Step 1 Decoder最终hidden误差超限。\n";

            return EXIT_FAILURE;
        }

        /*
         * Step 1 Final RMSNorm。
         */
        ncnn::Mat step1_norm_handoff;
        Metrics step1_norm_metrics;

        {
            ncnn::Net network;

            network.opt.use_vulkan_compute =
                false;

            network.opt.use_packing_layout =
                use_packing_layout;

            network.opt.num_threads =
                kRuntimeThreads;

            const std::string param_path =
                final_norm_directory
                + "/final_norm.ncnn.param";

            const std::string model_path =
                final_norm_directory
                + "/final_norm.ncnn.bin";

            if (
                network.load_param(
                    param_path.c_str()
                ) != 0
                || network.load_model(
                    model_path.c_str()
                ) != 0
            ) {
                std::cerr
                    << "Step 1 Final RMSNorm模型加载失败。\n";

                return EXIT_FAILURE;
            }

            ncnn::Extractor extractor =
                network.create_extractor();

            extractor.set_light_mode(false);

            if (
                extractor.input(
                    "in0",
                    step1_current_hidden
                ) != 0
            ) {
                std::cerr
                    << "Step 1 Final RMSNorm输入绑定失败。\n";

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
                    << "Step 1 Final RMSNorm输出提取失败。\n";

                return EXIT_FAILURE;
            }

            std::vector<float>
                actual_norm_output;

            if (
                !unpack_mat(
                    output,
                    actual_norm_output
                )
                || !calculate_metrics(
                    actual_norm_output,
                    expected_step1_norm_output,
                    step1_norm_metrics
                )
            ) {
                return EXIT_FAILURE;
            }

            step1_norm_handoff =
                output.clone();
        }

        if (step1_norm_handoff.empty()) {
            std::cerr
                << "Step 1 Final RMSNorm out0.clone()失败。\n";

            return EXIT_FAILURE;
        }

        std::cout
            << "\n===== Step 1 Final RMSNorm parity =====\n";

        print_metrics(
            "Step 1 Final RMSNorm output",
            step1_norm_metrics
        );

        const bool step1_norm_passed =
            step1_norm_metrics.maximum_abs_error
                <= 5.0e-5
            && step1_norm_metrics.mean_abs_error
                <= 5.0e-6
            && step1_norm_metrics.cosine_similarity
                >= 0.99999999;

        if (!step1_norm_passed) {
            std::cerr
                << "❌ Step 1 Final RMSNorm误差超限。\n";

            return EXIT_FAILURE;
        }

        /*
         * Step 1 LM Head并计算实际反馈token。
         */
        std::vector<float>
            actual_step1_logits;

        Metrics step1_logits_metrics;

        {
            ncnn::Net network;

            network.opt.use_vulkan_compute =
                false;

            network.opt.use_packing_layout =
                use_packing_layout;

            network.opt.num_threads =
                kRuntimeThreads;

            const std::string param_path =
                lm_head_directory
                + "/lm_head.ncnn.param";

            const std::string model_path =
                lm_head_directory
                + "/lm_head.ncnn.bin";

            if (
                network.load_param(
                    param_path.c_str()
                ) != 0
                || network.load_model(
                    model_path.c_str()
                ) != 0
            ) {
                std::cerr
                    << "Step 1 LM Head模型加载失败。\n";

                return EXIT_FAILURE;
            }

            ncnn::Extractor extractor =
                network.create_extractor();

            extractor.set_light_mode(false);

            if (
                extractor.input(
                    "in0",
                    step1_norm_handoff
                ) != 0
            ) {
                std::cerr
                    << "Step 1 LM Head输入绑定失败。\n";

                return EXIT_FAILURE;
            }

            ncnn::Mat logits;

            if (
                extractor.extract(
                    "out0",
                    logits
                ) != 0
                || logits.empty()
            ) {
                std::cerr
                    << "Step 1 LM Head输出提取失败。\n";

                return EXIT_FAILURE;
            }

            if (
                !unpack_mat(
                    logits,
                    actual_step1_logits
                )
            ) {
                return EXIT_FAILURE;
            }
        }

        if (
            actual_step1_logits.size()
            != kFeedbackVocabSize
        ) {
            std::cerr
                << "Step 1 logits数量错误："
                << actual_step1_logits.size()
                << '\n';

            return EXIT_FAILURE;
        }

        if (
            !calculate_metrics(
                actual_step1_logits,
                expected_step1_logits,
                step1_logits_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        const auto expected_iterator =
            std::max_element(
                expected_step1_logits.begin(),
                expected_step1_logits.end()
            );

        const auto actual_iterator =
            std::max_element(
                actual_step1_logits.begin(),
                actual_step1_logits.end()
            );

        const int step1_expected_token =
            static_cast<int>(
                std::distance(
                    expected_step1_logits.begin(),
                    expected_iterator
                )
            );

        step1_actual_token =
            static_cast<int>(
                std::distance(
                    actual_step1_logits.begin(),
                    actual_iterator
                )
            );

        std::cout
            << "\n===== Step 1 feedback logits parity =====\n";

        print_metrics(
            "Step 1 Decoder -> Final RMSNorm -> LM Head logits",
            step1_logits_metrics
        );

        std::cout
            << "Expected Step 1 token : "
            << step1_expected_token
            << '\n'
            << "Actual Step 1 token   : "
            << step1_actual_token
            << '\n'
            << "Feedback contract     : "
            << kExpectedStep1Token
            << '\n';

        const bool step1_logits_passed =
            step1_logits_metrics.maximum_abs_error
                <= 3.0e-3
            && step1_logits_metrics.mean_abs_error
                <= 5.0e-5
            && step1_logits_metrics.cosine_similarity
                >= 0.999999
            && step1_expected_token
                == kExpectedStep1Token
            && step1_actual_token
                == kExpectedStep1Token;

        if (!step1_logits_passed) {
            std::cerr
                << "❌ Step 1反馈logits或token验证失败。\n";

            return EXIT_FAILURE;
        }

        /*
         * actual token 5112
         * -> int32 ncnn Embed
         * -> Step 2初始hidden。
         */
        ncnn::Mat token_input(
            1,
            static_cast<std::size_t>(4u)
        );

        if (token_input.empty()) {
            std::cerr
                << "反馈token Mat创建失败。\n";

            return EXIT_FAILURE;
        }

        int* token_pointer =
            token_input;

        token_pointer[0] =
            step1_actual_token;

        {
            ncnn::Net network;

            network.opt.use_vulkan_compute =
                false;

            network.opt.use_packing_layout =
                use_packing_layout;

            network.opt.num_threads =
                kRuntimeThreads;

            const std::string param_path =
                embedding_directory
                + "/text_embedding.ncnn.param";

            const std::string model_path =
                embedding_directory
                + "/text_embedding.ncnn.bin";

            if (
                network.load_param(
                    param_path.c_str()
                ) != 0
                || network.load_model(
                    model_path.c_str()
                ) != 0
            ) {
                std::cerr
                    << "反馈Token Embedding模型加载失败。\n";

                return EXIT_FAILURE;
            }

            ncnn::Extractor extractor =
                network.create_extractor();

            extractor.set_light_mode(false);

            if (
                extractor.input(
                    "in0",
                    token_input
                ) != 0
            ) {
                std::cerr
                    << "反馈Token Embedding输入绑定失败。\n";

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
                    << "反馈Token Embedding输出提取失败。\n";

                return EXIT_FAILURE;
            }

            step2_current_hidden =
                output.clone();
        }

        if (step2_current_hidden.empty()) {
            std::cerr
                << "反馈Embedding out0.clone()失败。\n";

            return EXIT_FAILURE;
        }

        std::vector<float>
            generated_step2_initial_hidden;

        if (
            !unpack_mat(
                step2_current_hidden,
                generated_step2_initial_hidden
            )
        ) {
            return EXIT_FAILURE;
        }

        Metrics feedback_embedding_metrics;

        if (
            !calculate_metrics(
                generated_step2_initial_hidden,
                expected_step2_initial_hidden,
                feedback_embedding_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        const bool feedback_byte_identical =
            generated_step2_initial_hidden.size()
                == expected_step2_initial_hidden.size()
            && std::equal(
                generated_step2_initial_hidden.begin(),
                generated_step2_initial_hidden.end(),
                expected_step2_initial_hidden.begin()
            );

        std::cout
            << "\n===== Autoregressive Token Embedding feedback =====\n";

        print_metrics(
            "Step 1 token 5112 -> ncnn Embed -> Step 2 Layer 0 hidden",
            feedback_embedding_metrics
        );

        std::cout
            << "Feedback token        : "
            << step1_actual_token
            << '\n'
            << "Embedding input type  : int32\n"
            << "Reference used as input: false\n"
            << "Byte-identical hidden : "
            << (
                feedback_byte_identical
                ? "true"
                : "false"
            )
            << '\n';

        if (
            feedback_embedding_metrics.maximum_abs_error
                != 0.0
            || feedback_embedding_metrics.mean_abs_error
                != 0.0
            || !feedback_byte_identical
        ) {
            std::cerr
                << "❌ 自回归Embedding反馈hidden"
                << "未达到字节级一致。\n";

            return EXIT_FAILURE;
        }
    }

    std::cout
        << "\n===== Step 2: direct same-layer KV chain =====\n"
        << "Step 2 initial hidden source: "
        << "Step 1 actual token -> ncnn Embed\n"
        << "Step 2 initial hidden reload: disabled\n";

    for (
        int layer = 0;
        layer < kLayerCount;
        ++layer
    ) {
        const std::string layer_text =
            std::to_string(layer);

        const std::string name =
            "decoder_layer"
            + layer_text
            + "_decode_step2";

        const std::string model_directory =
            project_root
            + "/artifacts/"
            + name;

        const std::string reference =
            model_directory
            + "/reference";

        std::vector<float> expected_hidden;
        std::vector<float> expected_past_key;
        std::vector<float> expected_past_value;
        std::vector<float> expected_output;
        std::vector<float> expected_present_key;
        std::vector<float> expected_present_value;

        if (
            !load_exact_binary(
                reference
                    + "/layer"
                    + layer_text
                    + "_hidden_states_f32.bin",
                hidden_count,
                expected_hidden
            )
            || !load_exact_binary(
                reference
                    + "/past_key_f32.bin",
                step2_past_count,
                expected_past_key
            )
            || !load_exact_binary(
                reference
                    + "/past_value_f32.bin",
                step2_past_count,
                expected_past_value
            )
            || !load_exact_binary(
                reference
                    + "/layer"
                    + layer_text
                    + "_output_f32.bin",
                hidden_count,
                expected_output
            )
            || !load_exact_binary(
                reference
                    + "/present_key_f32.bin",
                step2_present_count,
                expected_present_key
            )
            || !load_exact_binary(
                reference
                    + "/present_value_f32.bin",
                step2_present_count,
                expected_present_value
            )
        ) {
            std::cerr
                << "Step 2 Layer "
                << layer
                << "参考数据加载失败。\n";

            return EXIT_FAILURE;
        }

        /*
         * expected_past_key/value只用于边界比较。
         * Step 2 ncnn in4/in5不会绑定这些参考数组。
         */
        std::vector<float> actual_input;
        std::vector<float> handoff_key_values;
        std::vector<float> handoff_value_values;

        if (
            !unpack_mat(
                step2_current_hidden,
                actual_input
            )
            || !unpack_mat(
                step1_present_keys[layer],
                handoff_key_values
            )
            || !unpack_mat(
                step1_present_values[layer],
                handoff_value_values
            )
        ) {
            return EXIT_FAILURE;
        }

        Metrics input_metrics;
        Metrics handoff_key_metrics;
        Metrics handoff_value_metrics;

        if (
            !calculate_metrics(
                actual_input,
                expected_hidden,
                input_metrics
            )
            || !calculate_metrics(
                handoff_key_values,
                expected_past_key,
                handoff_key_metrics
            )
            || !calculate_metrics(
                handoff_value_values,
                expected_past_value,
                handoff_value_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        ncnn::Mat next_hidden;
        ncnn::Mat present_key;
        ncnn::Mat present_value;

        {
            ncnn::Net network;

            network.opt.use_vulkan_compute =
                false;

            network.opt.use_packing_layout =
                use_packing_layout;

            network.opt.num_threads =
                kRuntimeThreads;

            const std::string param_path =
                model_directory
                + "/"
                + name
                + ".ncnn.param";

            const std::string model_path =
                model_directory
                + "/"
                + name
                + ".ncnn.bin";

            if (
                network.load_param(
                    param_path.c_str()
                ) != 0
                || network.load_model(
                    model_path.c_str()
                ) != 0
            ) {
                std::cerr
                    << "Step 2 Layer "
                    << layer
                    << "模型加载失败。\n";

                return EXIT_FAILURE;
            }

            ncnn::Extractor extractor =
                network.create_extractor();

            extractor.set_light_mode(false);

            if (
                extractor.input(
                    "in0",
                    step2_current_hidden
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
                    step1_present_keys[layer]
                ) != 0
                || extractor.input(
                    "in5",
                    step1_present_values[layer]
                ) != 0
            ) {
                std::cerr
                    << "Step 2 Layer "
                    << layer
                    << "输入绑定失败。\n";

                return EXIT_FAILURE;
            }

            ncnn::Mat output;

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
                    << "Step 2 Layer "
                    << layer
                    << "输出提取失败。\n";

                return EXIT_FAILURE;
            }

            next_hidden =
                output.clone();

            present_key =
                present_key.clone();

            present_value =
                present_value.clone();
        }

        if (
            next_hidden.empty()
            || present_key.empty()
            || present_value.empty()
        ) {
            std::cerr
                << "Step 2 Layer "
                << layer
                << "输出clone失败。\n";

            return EXIT_FAILURE;
        }

        std::vector<float> actual_output;
        std::vector<float> actual_key;
        std::vector<float> actual_value;

        if (
            !unpack_mat(
                next_hidden,
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

        Metrics output_metrics;
        Metrics key_metrics;
        Metrics value_metrics;

        if (
            !calculate_metrics(
                actual_output,
                expected_output,
                output_metrics
            )
            || !calculate_metrics(
                actual_key,
                expected_present_key,
                key_metrics
            )
            || !calculate_metrics(
                actual_value,
                expected_present_value,
                value_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        const double key_prefix_error =
            cache_region_max_error(
                actual_key,
                kStep2PresentLength,
                handoff_key_values,
                kStep2PastLength,
                0,
                kStep2PastLength
            );

        const double value_prefix_error =
            cache_region_max_error(
                actual_value,
                kStep2PresentLength,
                handoff_value_values,
                kStep2PastLength,
                0,
                kStep2PastLength
            );

        const double appended_key_error =
            cache_region_max_error(
                actual_key,
                kStep2PresentLength,
                expected_present_key,
                kStep2PresentLength,
                kStep2PastLength,
                kStep2PresentLength
            );

        const double appended_value_error =
            cache_region_max_error(
                actual_value,
                kStep2PresentLength,
                expected_present_value,
                kStep2PresentLength,
                kStep2PastLength,
                kStep2PresentLength
            );

        step2_maximum_boundary_error =
            std::max(
                step2_maximum_boundary_error,
                input_metrics.maximum_abs_error
            );

        step2_maximum_output_error =
            std::max(
                step2_maximum_output_error,
                output_metrics.maximum_abs_error
            );

        step2_maximum_key_error =
            std::max(
                step2_maximum_key_error,
                key_metrics.maximum_abs_error
            );

        step2_maximum_value_error =
            std::max(
                step2_maximum_value_error,
                value_metrics.maximum_abs_error
            );

        maximum_handoff_key_error =
            std::max(
                maximum_handoff_key_error,
                handoff_key_metrics.maximum_abs_error
            );

        maximum_handoff_value_error =
            std::max(
                maximum_handoff_value_error,
                handoff_value_metrics.maximum_abs_error
            );

        maximum_step2_key_prefix_error =
            std::max(
                maximum_step2_key_prefix_error,
                key_prefix_error
            );

        maximum_step2_value_prefix_error =
            std::max(
                maximum_step2_value_prefix_error,
                value_prefix_error
            );

        maximum_step2_appended_key_error =
            std::max(
                maximum_step2_appended_key_error,
                appended_key_error
            );

        maximum_step2_appended_value_error =
            std::max(
                maximum_step2_appended_value_error,
                appended_value_error
            );

        std::cout
            << "Step2 Layer "
            << std::setw(2)
            << std::setfill('0')
            << layer
            << std::setfill(' ')
            << ": boundary="
            << std::scientific
            << std::setprecision(10)
            << input_metrics.maximum_abs_error
            << ", handoff=("
            << handoff_key_metrics.maximum_abs_error
            << ", "
            << handoff_value_metrics.maximum_abs_error
            << "), output="
            << output_metrics.maximum_abs_error
            << ", key="
            << key_metrics.maximum_abs_error
            << ", value="
            << value_metrics.maximum_abs_error
            << ", prefix=("
            << key_prefix_error
            << ", "
            << value_prefix_error
            << "), appended=("
            << appended_key_error
            << ", "
            << appended_value_error
            << ")\n";

        const bool passed =
            hidden_passed(input_metrics)
            && hidden_passed(output_metrics)
            && cache_passed(handoff_key_metrics)
            && cache_passed(handoff_value_metrics)
            && cache_passed(key_metrics)
            && cache_passed(value_metrics)
            && key_prefix_error == 0.0
            && value_prefix_error == 0.0
            && appended_key_error <= 2.0e-4
            && appended_value_error <= 2.0e-4;

        if (!passed) {
            std::cerr
                << "❌ Step 2 Layer "
                << layer
                << "数值验证失败。\n";

            print_metrics(
                "input boundary",
                input_metrics
            );

            print_metrics(
                "handoff key",
                handoff_key_metrics
            );

            print_metrics(
                "handoff value",
                handoff_value_metrics
            );

            print_metrics(
                "layer output",
                output_metrics
            );

            print_metrics(
                "present key",
                key_metrics
            );

            print_metrics(
                "present value",
                value_metrics
            );

            return EXIT_FAILURE;
        }

        step2_current_hidden =
            next_hidden;
    }

    std::cout
        << '\n'
        << std::scientific
        << std::setprecision(10)
        << "===== Two-step chain maximum errors =====\n"
        << "Step 1 maximum input boundary : "
        << step1_maximum_boundary_error
        << '\n'
        << "Step 1 maximum layer output   : "
        << step1_maximum_output_error
        << '\n'
        << "Step 1 maximum present key    : "
        << step1_maximum_key_error
        << '\n'
        << "Step 1 maximum present value  : "
        << step1_maximum_value_error
        << '\n'
        << "Step 2 maximum input boundary : "
        << step2_maximum_boundary_error
        << '\n'
        << "Step 2 maximum layer output   : "
        << step2_maximum_output_error
        << '\n'
        << "Step 2 maximum present key    : "
        << step2_maximum_key_error
        << '\n'
        << "Step 2 maximum present value  : "
        << step2_maximum_value_error
        << '\n'
        << "Maximum KV handoff key error  : "
        << maximum_handoff_key_error
        << '\n'
        << "Maximum KV handoff value error: "
        << maximum_handoff_value_error
        << '\n'
        << "Maximum Step 2 key prefix     : "
        << maximum_step2_key_prefix_error
        << '\n'
        << "Maximum Step 2 value prefix   : "
        << maximum_step2_value_prefix_error
        << '\n'
        << "Maximum Step 2 appended key   : "
        << maximum_step2_appended_key_error
        << '\n'
        << "Maximum Step 2 appended value : "
        << maximum_step2_appended_value_error
        << '\n';

    if (
        maximum_step2_key_prefix_error
            != 0.0
        || maximum_step2_value_prefix_error
            != 0.0
    ) {
        std::cerr
            << "\n❌ 两步链KV历史前缀发生变化。\n";

        return EXIT_FAILURE;
    }

    /*
     * Step 2 Decoder Layer 23 out0
     *     -> Final RMSNorm
     *     -> LM Head
     *     -> logits
     *
     * step2_current_hidden保持为真实链式输出，
     * 不用参考hidden替换。
     */
    constexpr std::size_t kDecodeVocabSize =
        120818;

    constexpr int kExpectedStep2Token =
        206;

    const std::string step2_tail_reference =
        project_root
        + "/artifacts/decode_tail_step2/reference";

    const std::string final_norm_directory =
        project_root
        + "/artifacts/final_norm";

    const std::string lm_head_directory =
        project_root
        + "/artifacts/lm_head";

    std::vector<float> expected_norm_input;
    std::vector<float> expected_norm_output;
    std::vector<float> expected_logits;

    if (
        !load_exact_binary(
            step2_tail_reference
                + "/final_norm_input_f32.bin",
            hidden_count,
            expected_norm_input
        )
        || !load_exact_binary(
            step2_tail_reference
                + "/final_norm_output_f32.bin",
            hidden_count,
            expected_norm_output
        )
        || !load_exact_binary(
            step2_tail_reference
                + "/decode_logits_f32.bin",
            kDecodeVocabSize,
            expected_logits
        )
    ) {
        std::cerr
            << "Step 2尾部参考数据加载失败。\n";

        return EXIT_FAILURE;
    }

    std::vector<float> actual_decoder_output;

    if (
        !unpack_mat(
            step2_current_hidden,
            actual_decoder_output
        )
    ) {
        return EXIT_FAILURE;
    }

    Metrics final_norm_input_metrics;

    if (
        !calculate_metrics(
            actual_decoder_output,
            expected_norm_input,
            final_norm_input_metrics
        )
    ) {
        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Step 2 Decoder -> Final RMSNorm boundary =====\n";

    print_metrics(
        "Step 2 Layer 23 chained output vs Final RMSNorm reference input",
        final_norm_input_metrics
    );

    if (
        !hidden_passed(
            final_norm_input_metrics
        )
    ) {
        std::cerr
            << "❌ Step 2 Decoder最终输出累计误差超限。\n";

        return EXIT_FAILURE;
    }

    ncnn::Mat norm_handoff;
    Metrics norm_metrics;

    {
        ncnn::Net final_norm_network;

        final_norm_network.opt.use_vulkan_compute =
            false;

        final_norm_network.opt.use_packing_layout =
            use_packing_layout;

        final_norm_network.opt.num_threads =
            kRuntimeThreads;

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
            || final_norm_network.load_model(
                model_path.c_str()
            ) != 0
        ) {
            std::cerr
                << "Final RMSNorm模型加载失败。\n";

            return EXIT_FAILURE;
        }

        ncnn::Extractor extractor =
            final_norm_network.create_extractor();

        extractor.set_light_mode(false);

        if (
            extractor.input(
                "in0",
                step2_current_hidden
            ) != 0
        ) {
            std::cerr
                << "Final RMSNorm输入绑定失败。\n";

            return EXIT_FAILURE;
        }

        ncnn::Mat norm_output;

        if (
            extractor.extract(
                "out0",
                norm_output
            ) != 0
            || norm_output.empty()
        ) {
            std::cerr
                << "Final RMSNorm输出提取失败。\n";

            return EXIT_FAILURE;
        }

        std::vector<float> actual_norm_output;

        if (
            !unpack_mat(
                norm_output,
                actual_norm_output
            )
            || !calculate_metrics(
                actual_norm_output,
                expected_norm_output,
                norm_metrics
            )
        ) {
            return EXIT_FAILURE;
        }

        /*
         * Final RMSNorm网络释放之前必须clone，
         * 防止LM Head引用已经失效的内存。
         */
        norm_handoff =
            norm_output.clone();
    }

    if (norm_handoff.empty()) {
        std::cerr
            << "Final RMSNorm out0.clone()失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Step 2 Final RMSNorm parity =====\n";

    print_metrics(
        "Step 2 Final RMSNorm output",
        norm_metrics
    );

    const bool norm_passed =
        norm_metrics.maximum_abs_error
            <= 5.0e-5
        && norm_metrics.mean_abs_error
            <= 5.0e-6
        && norm_metrics.cosine_similarity
            >= 0.99999999;

    if (!norm_passed) {
        std::cerr
            << "❌ Step 2 Final RMSNorm误差超限。\n";

        return EXIT_FAILURE;
    }

    std::vector<float> actual_logits;
    Metrics logits_metrics;

    {
        ncnn::Net lm_head_network;

        lm_head_network.opt.use_vulkan_compute =
            false;

        lm_head_network.opt.use_packing_layout =
            use_packing_layout;

        lm_head_network.opt.num_threads =
            kRuntimeThreads;

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
            || lm_head_network.load_model(
                model_path.c_str()
            ) != 0
        ) {
            std::cerr
                << "LM Head模型加载失败。\n";

            return EXIT_FAILURE;
        }

        ncnn::Extractor extractor =
            lm_head_network.create_extractor();

        extractor.set_light_mode(false);

        if (
            extractor.input(
                "in0",
                norm_handoff
            ) != 0
        ) {
            std::cerr
                << "LM Head输入绑定失败。\n";

            return EXIT_FAILURE;
        }

        ncnn::Mat logits;

        if (
            extractor.extract(
                "out0",
                logits
            ) != 0
            || logits.empty()
        ) {
            std::cerr
                << "LM Head输出提取失败。\n";

            return EXIT_FAILURE;
        }

        if (
            !unpack_mat(
                logits,
                actual_logits
            )
        ) {
            return EXIT_FAILURE;
        }
    }

    if (
        actual_logits.size()
        != kDecodeVocabSize
    ) {
        std::cerr
            << "Step 2 logits数量错误："
            << actual_logits.size()
            << "，预期："
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
        << "\n===== Step 2 full logits parity =====\n";

    print_metrics(
        "Two-step Decoder -> Final RMSNorm -> LM Head logits",
        logits_metrics
    );

    std::cout
        << "Expected Step 2 token : "
        << expected_token
        << '\n'
        << "Actual Step 2 token   : "
        << actual_token
        << '\n'
        << "Contract token        : "
        << kExpectedStep2Token
        << '\n';

    const bool logits_passed =
        logits_metrics.maximum_abs_error
            <= 3.0e-3
        && logits_metrics.mean_abs_error
            <= 5.0e-5
        && logits_metrics.cosine_similarity
            >= 0.999999
        && expected_token
            == kExpectedStep2Token
        && actual_token
            == kExpectedStep2Token;

    if (!logits_passed) {
        std::cerr
            << "❌ Step 2完整logits数值验证失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Two-step logits-chain result =====\n"
        << "Step 1 decoder layers : 0 -> 23\n"
        << "Step 2 decoder layers : 0 -> 23\n"
        << "Same-layer KV handoff : enabled\n"
        << "Step 1 feedback token  : "
        << step1_actual_token
        << '\n'
        << "Token Embedding        : executed\n"
        << "Step 2 hidden reload   : disabled\n"
        << "Final RMSNorm          : executed\n"
        << "LM Head                : executed\n"
        << "Tail input reload      : disabled\n"
        << "Final Step 2 token     : "
        << actual_token
        << '\n';

    std::cout
        << "\n✅ 两次Decode的24层Decoder"
        << " → Final RMSNorm"
        << " → LM Head"
        << " → token 206"
        << " 完整ncnn FP32链验证成功。\n";

    return EXIT_SUCCESS;
}
