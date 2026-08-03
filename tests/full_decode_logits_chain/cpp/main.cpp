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
    if (argc != 5) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <project_root>"
            << " <start_layer>"
            << " <end_layer>"
            << " <packing:0|1>\n";

        return EXIT_FAILURE;
    }

    const std::string project_root = argv[1];

    auto parse_layer = [](
        const char* text,
        int& result)
    {
        char* end = nullptr;

        const long value = std::strtol(
            text,
            &end,
            10
        );

        if (
            end == text
            || *end != '\0'
            || value < 0
            || value > 23
        ) {
            return false;
        }

        result = static_cast<int>(value);
        return true;
    };

    int start_layer = -1;
    int end_layer = -1;

    if (
        !parse_layer(argv[2], start_layer)
        || !parse_layer(argv[3], end_layer)
        || start_layer > end_layer
    ) {
        std::cerr
            << "层范围无效："
            << argv[2]
            << " -> "
            << argv[3]
            << '\n';

        return EXIT_FAILURE;
    }

    const std::string packing_text = argv[4];

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

    auto model_directory =
        [&project_root](
            const int layer)
        {
            return
                project_root
                + "/artifacts/decoder_layer"
                + std::to_string(layer)
                + "_decode";
        };

    auto reference_directory =
        [&model_directory](
            const int layer)
        {
            return
                model_directory(layer)
                + "/reference";
        };

    auto layer_prefix = [](
        const int layer)
        {
            return
                "layer"
                + std::to_string(layer);
        };

    /*
     * 全局参考链审计已经证明所有Decoder层使用的
     * attention mask和RoPE张量逐元素完全相同。
     * 因此这里只加载Layer 0的一份公共输入。
     */
    const std::string common_reference =
        reference_directory(0);

    std::vector<float> mask_values;
    std::vector<float> rope_cos_values;
    std::vector<float> rope_sin_values;

    if (
        !load_exact_binary(
            common_reference
                + "/layer0_attention_mask_f32.bin",
            kMaskCount,
            mask_values
        )
        || !load_exact_binary(
            common_reference
                + "/layer0_position_embeddings_0_f32.bin",
            kRopeCount,
            rope_cos_values
        )
        || !load_exact_binary(
            common_reference
                + "/layer0_position_embeddings_1_f32.bin",
            kRopeCount,
            rope_sin_values
        )
    ) {
        return EXIT_FAILURE;
    }

    auto make_hidden = [](
        std::vector<float>& values)
    {
        ncnn::Mat flat(
            static_cast<int>(kHiddenCount),
            static_cast<void*>(values.data())
        );

        return flat.reshape(
            kHiddenSize,
            1
        ).clone();
    };

    auto make_mask = [](
        std::vector<float>& values)
    {
        ncnn::Mat flat(
            static_cast<int>(kMaskCount),
            static_cast<void*>(values.data())
        );

        return flat.reshape(
            kMaskLength,
            1,
            1
        ).clone();
    };

    auto make_rope = [](
        std::vector<float>& values)
    {
        ncnn::Mat flat(
            static_cast<int>(kRopeCount),
            static_cast<void*>(values.data())
        );

        return flat.reshape(
            kHeadDim,
            1,
            kRopeComponents
        ).clone();
    };

    auto make_past_cache = [](
        std::vector<float>& values)
    {
        ncnn::Mat flat(
            static_cast<int>(kPastCacheCount),
            static_cast<void*>(values.data())
        );

        return flat.reshape(
            kHeadDim,
            kPastLength,
            kKeyValueHeads
        ).clone();
    };

    ncnn::Mat attention_mask =
        make_mask(mask_values);

    ncnn::Mat rope_cos =
        make_rope(rope_cos_values);

    ncnn::Mat rope_sin =
        make_rope(rope_sin_values);

    if (
        attention_mask.empty()
        || rope_cos.empty()
        || rope_sin.empty()
    ) {
        std::cerr
            << "创建公共Mask或RoPE Mat失败。\n";

        return EXIT_FAILURE;
    }

    /*
     * 只从起始层的参考文件加载一次推理hidden。
     * 后续所有层都使用前一层ncnn out0作为输入。
     */
    const std::string initial_prefix =
        layer_prefix(start_layer);

    std::vector<float> initial_hidden_values;

    if (
        !load_exact_binary(
            reference_directory(start_layer)
                + "/"
                + initial_prefix
                + "_hidden_states_f32.bin",
            kHiddenCount,
            initial_hidden_values
        )
    ) {
        return EXIT_FAILURE;
    }

    ncnn::Mat current_hidden =
        make_hidden(initial_hidden_values);

    if (current_hidden.empty()) {
        std::cerr
            << "创建初始hidden Mat失败。\n";

        return EXIT_FAILURE;
    }

    auto cache_region_error = [](
        const std::vector<float>& actual,
        const std::vector<float>& expected,
        const int begin_token,
        const int end_token)
    {
        double maximum_error = 0.0;

        for (
            int head = 0;
            head < kKeyValueHeads;
            ++head
        ) {
            for (
                int token = begin_token;
                token < end_token;
                ++token
            ) {
                for (
                    int dimension = 0;
                    dimension < kHeadDim;
                    ++dimension
                ) {
                    const std::size_t index =
                        (
                            (
                                static_cast<std::size_t>(head)
                                * static_cast<std::size_t>(
                                    kPresentLength
                                )
                                + static_cast<std::size_t>(
                                    token
                                )
                            )
                            * static_cast<std::size_t>(
                                kHeadDim
                            )
                            + static_cast<std::size_t>(
                                dimension
                            )
                        );

                    const double error = std::fabs(
                        static_cast<double>(
                            actual[index]
                        )
                        - static_cast<double>(
                            expected[index]
                        )
                    );

                    maximum_error = std::max(
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
            metrics.maximum_abs_error <= 2.0e-3
            && metrics.mean_abs_error <= 1.0e-5
            && metrics.cosine_similarity
                >= 0.99999999;
    };

    auto cache_passed = [](
        const Metrics& metrics)
    {
        return
            metrics.maximum_abs_error <= 2.0e-4
            && metrics.mean_abs_error <= 1.0e-6
            && metrics.cosine_similarity
                >= 0.99999999;
    };

    double maximum_boundary_error = 0.0;
    double maximum_output_error = 0.0;
    double maximum_key_error = 0.0;
    double maximum_value_error = 0.0;

    std::cout
        << "===== N-layer runtime options =====\n"
        << "project root        : "
        << project_root
        << '\n'
        << "start layer         : "
        << start_layer
        << '\n'
        << "end layer           : "
        << end_layer
        << '\n'
        << "layer count         : "
        << end_layer - start_layer + 1
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
        << "handoff              : "
        << "previous out0.clone() -> next in0\n";

    std::cout
        << "\n===== Initial hidden shape =====\n";

    print_shape(
        "current_hidden",
        current_hidden
    );

    std::cout
        << "\n===== Per-layer numerical summary =====\n"
        << "layer  input_max         output_max        "
        << "key_max           value_max         "
        << "key_prefix       value_prefix\n"
        << std::scientific
        << std::setprecision(10);

    for (
        int layer = start_layer;
        layer <= end_layer;
        ++layer
    ) {
        const std::string name =
            "decoder_layer"
            + std::to_string(layer)
            + "_decode";

        const std::string directory =
            model_directory(layer);

        const std::string reference =
            reference_directory(layer);

        const std::string prefix =
            layer_prefix(layer);

        std::vector<float> expected_hidden;
        std::vector<float> expected_output;
        std::vector<float> expected_present_key;
        std::vector<float> expected_present_value;

        std::vector<float> past_key_values;
        std::vector<float> past_value_values;

        if (
            !load_exact_binary(
                reference
                    + "/"
                    + prefix
                    + "_hidden_states_f32.bin",
                kHiddenCount,
                expected_hidden
            )
            || !load_exact_binary(
                reference
                    + "/"
                    + prefix
                    + "_output_f32.bin",
                kHiddenCount,
                expected_output
            )
            || !load_exact_binary(
                reference + "/past_key_f32.bin",
                kPastCacheCount,
                past_key_values
            )
            || !load_exact_binary(
                reference + "/past_value_f32.bin",
                kPastCacheCount,
                past_value_values
            )
            || !load_exact_binary(
                reference + "/present_key_f32.bin",
                kPresentCacheCount,
                expected_present_key
            )
            || !load_exact_binary(
                reference + "/present_value_f32.bin",
                kPresentCacheCount,
                expected_present_value
            )
        ) {
            std::cerr
                << "Layer "
                << layer
                << "参考数据加载失败。\n";

            return EXIT_FAILURE;
        }

        /*
         * expected_hidden只参与数值检查，
         * 不会被转换成下一层的推理输入。
         */
        std::vector<float> actual_input;

        if (
            !unpack_mat(
                current_hidden,
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
            make_past_cache(past_key_values);

        ncnn::Mat past_value =
            make_past_cache(past_value_values);

        if (
            past_key.empty()
            || past_value.empty()
        ) {
            std::cerr
                << "Layer "
                << layer
                << " KV Cache Mat创建失败。\n";

            return EXIT_FAILURE;
        }

        Metrics output_metrics;
        Metrics key_metrics;
        Metrics value_metrics;

        double key_prefix_error = 0.0;
        double value_prefix_error = 0.0;

        ncnn::Mat next_hidden;

        /*
         * 每轮只保留一个Decoder Layer网络。
         * 离开此作用域后当前network会被释放。
         */
        {
            ncnn::Net network;

            network.opt.use_vulkan_compute = false;
            network.opt.use_packing_layout =
                use_packing_layout;
            network.opt.num_threads = kThreads;

            const std::string param_path =
                directory
                + "/"
                + name
                + ".ncnn.param";

            const std::string model_path =
                directory
                + "/"
                + name
                + ".ncnn.bin";

            if (
                network.load_param(
                    param_path.c_str()
                ) != 0
            ) {
                std::cerr
                    << "Layer "
                    << layer
                    << " param加载失败："
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
                    << "Layer "
                    << layer
                    << " bin加载失败："
                    << model_path
                    << '\n';

                return EXIT_FAILURE;
            }

            ncnn::Extractor extractor =
                network.create_extractor();

            extractor.set_light_mode(false);

            if (
                extractor.input(
                    "in0",
                    current_hidden
                ) != 0
                || extractor.input(
                    "in1",
                    attention_mask
                ) != 0
                || extractor.input(
                    "in2",
                    rope_cos
                ) != 0
                || extractor.input(
                    "in3",
                    rope_sin
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
                    << "Layer "
                    << layer
                    << "输入绑定失败。\n";

                return EXIT_FAILURE;
            }

            ncnn::Mat output;
            ncnn::Mat present_key;
            ncnn::Mat present_value;

            const int output_status =
                extractor.extract(
                    "out0",
                    output
                );

            const int key_status =
                extractor.extract(
                    "out1",
                    present_key
                );

            const int value_status =
                extractor.extract(
                    "out2",
                    present_value
                );

            if (
                output_status != 0
                || key_status != 0
                || value_status != 0
            ) {
                std::cerr
                    << "Layer "
                    << layer
                    << "输出提取失败："
                    << "out0="
                    << output_status
                    << "，out1="
                    << key_status
                    << "，out2="
                    << value_status
                    << '\n';

                return EXIT_FAILURE;
            }

            std::vector<float> actual_output;
            std::vector<float> actual_present_key;
            std::vector<float> actual_present_value;

            if (
                !unpack_mat(
                    output,
                    actual_output
                )
                || !unpack_mat(
                    present_key,
                    actual_present_key
                )
                || !unpack_mat(
                    present_value,
                    actual_present_value
                )
            ) {
                return EXIT_FAILURE;
            }

            if (
                !calculate_metrics(
                    actual_output,
                    expected_output,
                    output_metrics
                )
                || !calculate_metrics(
                    actual_present_key,
                    expected_present_key,
                    key_metrics
                )
                || !calculate_metrics(
                    actual_present_value,
                    expected_present_value,
                    value_metrics
                )
            ) {
                return EXIT_FAILURE;
            }

            key_prefix_error =
                cache_region_error(
                    actual_present_key,
                    expected_present_key,
                    0,
                    kPastLength
                );

            value_prefix_error =
                cache_region_error(
                    actual_present_value,
                    expected_present_value,
                    0,
                    kPastLength
                );

            /*
             * 必须在network和extractor销毁前复制out0。
             * 该独立Mat会成为下一层的真实输入。
             */
            next_hidden = output.clone();

            if (next_hidden.empty()) {
                std::cerr
                    << "Layer "
                    << layer
                    << " out0.clone()失败。\n";

                return EXIT_FAILURE;
            }
        }

        maximum_boundary_error = std::max(
            maximum_boundary_error,
            input_metrics.maximum_abs_error
        );

        maximum_output_error = std::max(
            maximum_output_error,
            output_metrics.maximum_abs_error
        );

        maximum_key_error = std::max(
            maximum_key_error,
            key_metrics.maximum_abs_error
        );

        maximum_value_error = std::max(
            maximum_value_error,
            value_metrics.maximum_abs_error
        );

        std::cout
            << std::setw(5)
            << layer
            << "  "
            << std::setw(16)
            << input_metrics.maximum_abs_error
            << "  "
            << std::setw(16)
            << output_metrics.maximum_abs_error
            << "  "
            << std::setw(16)
            << key_metrics.maximum_abs_error
            << "  "
            << std::setw(16)
            << value_metrics.maximum_abs_error
            << "  "
            << std::setw(16)
            << key_prefix_error
            << "  "
            << std::setw(16)
            << value_prefix_error
            << '\n';

        const bool layer_passed =
            hidden_passed(input_metrics)
            && hidden_passed(output_metrics)
            && cache_passed(key_metrics)
            && cache_passed(value_metrics)
            && key_prefix_error == 0.0
            && value_prefix_error == 0.0;

        if (!layer_passed) {
            std::cerr
                << "\n❌ Layer "
                << layer
                << "累计串联误差超限。\n";

            print_metrics(
                "layer input boundary",
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

        /*
         * 不读取下一层的参考hidden作为推理输入。
         */
        current_hidden = next_hidden;
    }

    std::cout
        << "\n===== Decoder final hidden shape =====\n";

    print_shape(
        "decoder_final_hidden",
        current_hidden
    );

    std::cout
        << '\n'
        << std::scientific
        << std::setprecision(10)
        << "===== Decoder chain maximum errors =====\n"
        << "Maximum input boundary error : "
        << maximum_boundary_error
        << '\n'
        << "Maximum layer output error   : "
        << maximum_output_error
        << '\n'
        << "Maximum present-key error    : "
        << maximum_key_error
        << '\n'
        << "Maximum present-value error  : "
        << maximum_value_error
        << '\n';

    /*
     * 从这里开始进入真实Decode尾部：
     *
     * Decoder Layer 23 out0
     *     -> Final RMSNorm
     *     -> LM Head
     *     -> logits
     *
     * 不使用参考hidden替换current_hidden。
     */
    constexpr std::size_t kDecodeVocabSize =
        120818;

    constexpr int kExpectedDecodeToken =
        5112;

    const std::string decode_tail_reference =
        project_root
        + "/artifacts/decode_tail/reference";

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
            decode_tail_reference
                + "/final_norm_input_f32.bin",
            kHiddenCount,
            expected_norm_input
        )
        || !load_exact_binary(
            decode_tail_reference
                + "/final_norm_output_f32.bin",
            kHiddenCount,
            expected_norm_output
        )
        || !load_exact_binary(
            decode_tail_reference
                + "/decode_logits_f32.bin",
            kDecodeVocabSize,
            expected_logits
        )
    ) {
        return EXIT_FAILURE;
    }

    std::vector<float> actual_decoder_output;

    if (
        !unpack_mat(
            current_hidden,
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
        << "\n===== Decoder -> Final RMSNorm boundary =====\n";

    print_metrics(
        "Layer 23 chained output vs Final RMSNorm reference input",
        final_norm_input_metrics
    );

    if (
        !hidden_passed(
            final_norm_input_metrics
        )
    ) {
        std::cerr
            << "❌ Decoder最终输出累计误差超限。\n";

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
            << "\n===== Final RMSNorm output shape =====\n";

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
         * 在Final RMSNorm网络释放之前，
         * 将输出复制为独立拥有内存的ncnn::Mat。
         */
        norm_handoff =
            norm_output.clone();

        if (norm_handoff.empty()) {
            std::cerr
                << "Final RMSNorm out0.clone()失败。\n";

            return EXIT_FAILURE;
        }
    }

    std::cout
        << "\n===== Full-chain Final RMSNorm parity =====\n";

    print_metrics(
        "Final RMSNorm output",
        norm_metrics
    );

    /*
     * 独立tail测试测得：
     * max  = 1.9073486328e-06
     * mean = 3.0571140996e-07
     *
     * 完整24层输入包含约1e-7级累计误差，
     * 因此为完整链保留合理但严格的余量。
     */
    const bool norm_passed =
        norm_metrics.maximum_abs_error
            <= 5.0e-5
        && norm_metrics.mean_abs_error
            <= 5.0e-6
        && norm_metrics.cosine_similarity
            >= 0.99999999;

    if (!norm_passed) {
        std::cerr
            << "❌ 完整链Final RMSNorm误差超限。\n";

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
            << "\n===== Full decode logits shape =====\n";

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
                << "Logits元素数量错误："
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
        << "\n===== Full decode logits parity =====\n";

    print_metrics(
        "Decoder 0-23 -> Final RMSNorm -> LM Head logits",
        logits_metrics
    );

    std::cout
        << "Expected decode token : "
        << expected_token
        << '\n'
        << "Actual decode token   : "
        << actual_token
        << '\n'
        << "Contract token        : "
        << kExpectedDecodeToken
        << '\n';

    const bool logits_passed =
        logits_metrics.maximum_abs_error
            <= 3.0e-3
        && logits_metrics.mean_abs_error
            <= 5.0e-5
        && logits_metrics.cosine_similarity
            >= 0.999999
        && expected_token
            == kExpectedDecodeToken
        && actual_token
            == kExpectedDecodeToken;

    if (!logits_passed) {
        std::cerr
            << "❌ 完整Decode logits数值验证失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Full-chain result =====\n"
        << "Decoder layers       : 0 -> 23\n"
        << "Final RMSNorm        : executed\n"
        << "LM Head              : executed\n"
        << "Intermediate reload  : disabled\n"
        << "Final decode token   : "
        << actual_token
        << '\n';

    std::cout
        << "\n✅ Decoder Layer 0 → 23"
        << " → Final RMSNorm"
        << " → LM Head"
        << " → logits"
        << " 完整ncnn FP32 Decode链验证成功。\n";

    return EXIT_SUCCESS;
}
