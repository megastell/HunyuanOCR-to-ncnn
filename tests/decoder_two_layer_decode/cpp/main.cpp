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
    if (argc != 21) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <layer0.ncnn.param>"
            << " <layer0.ncnn.bin>"
            << " <layer1.ncnn.param>"
            << " <layer1.ncnn.bin>"
            << " <layer0_hidden_f32.bin>"
            << " <attention_mask_f32.bin>"
            << " <rope_cos_f32.bin>"
            << " <rope_sin_f32.bin>"
            << " <layer0_past_key_f32.bin>"
            << " <layer0_past_value_f32.bin>"
            << " <layer1_past_key_f32.bin>"
            << " <layer1_past_value_f32.bin>"
            << " <expected_layer0_output_f32.bin>"
            << " <expected_layer0_present_key_f32.bin>"
            << " <expected_layer0_present_value_f32.bin>"
            << " <expected_layer1_hidden_f32.bin>"
            << " <expected_layer1_output_f32.bin>"
            << " <expected_layer1_present_key_f32.bin>"
            << " <expected_layer1_present_value_f32.bin>"
            << " <packing:0|1>\n";

        return EXIT_FAILURE;
    }

    const std::string layer0_param_path = argv[1];
    const std::string layer0_model_path = argv[2];
    const std::string layer1_param_path = argv[3];
    const std::string layer1_model_path = argv[4];

    const bool use_packing_layout =
        std::string(argv[20]) == "1";

    std::vector<float> layer0_hidden_values;
    std::vector<float> mask_values;
    std::vector<float> rope_cos_values;
    std::vector<float> rope_sin_values;

    std::vector<float> layer0_past_key_values;
    std::vector<float> layer0_past_value_values;
    std::vector<float> layer1_past_key_values;
    std::vector<float> layer1_past_value_values;

    std::vector<float> expected_layer0_output;
    std::vector<float> expected_layer0_present_key;
    std::vector<float> expected_layer0_present_value;

    std::vector<float> expected_layer1_hidden;
    std::vector<float> expected_layer1_output;
    std::vector<float> expected_layer1_present_key;
    std::vector<float> expected_layer1_present_value;

    if (
        !load_exact_binary(
            argv[5],
            kHiddenCount,
            layer0_hidden_values
        )
        || !load_exact_binary(
            argv[6],
            kMaskCount,
            mask_values
        )
        || !load_exact_binary(
            argv[7],
            kRopeCount,
            rope_cos_values
        )
        || !load_exact_binary(
            argv[8],
            kRopeCount,
            rope_sin_values
        )
        || !load_exact_binary(
            argv[9],
            kPastCacheCount,
            layer0_past_key_values
        )
        || !load_exact_binary(
            argv[10],
            kPastCacheCount,
            layer0_past_value_values
        )
        || !load_exact_binary(
            argv[11],
            kPastCacheCount,
            layer1_past_key_values
        )
        || !load_exact_binary(
            argv[12],
            kPastCacheCount,
            layer1_past_value_values
        )
        || !load_exact_binary(
            argv[13],
            kHiddenCount,
            expected_layer0_output
        )
        || !load_exact_binary(
            argv[14],
            kPresentCacheCount,
            expected_layer0_present_key
        )
        || !load_exact_binary(
            argv[15],
            kPresentCacheCount,
            expected_layer0_present_value
        )
        || !load_exact_binary(
            argv[16],
            kHiddenCount,
            expected_layer1_hidden
        )
        || !load_exact_binary(
            argv[17],
            kHiddenCount,
            expected_layer1_output
        )
        || !load_exact_binary(
            argv[18],
            kPresentCacheCount,
            expected_layer1_present_key
        )
        || !load_exact_binary(
            argv[19],
            kPresentCacheCount,
            expected_layer1_present_value
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

    ncnn::Mat layer0_hidden =
        make_hidden(layer0_hidden_values);

    ncnn::Mat attention_mask =
        make_mask(mask_values);

    ncnn::Mat rope_cos =
        make_rope(rope_cos_values);

    ncnn::Mat rope_sin =
        make_rope(rope_sin_values);

    ncnn::Mat layer0_past_key =
        make_past_cache(
            layer0_past_key_values
        );

    ncnn::Mat layer0_past_value =
        make_past_cache(
            layer0_past_value_values
        );

    ncnn::Mat layer1_past_key =
        make_past_cache(
            layer1_past_key_values
        );

    ncnn::Mat layer1_past_value =
        make_past_cache(
            layer1_past_value_values
        );

    if (
        layer0_hidden.empty()
        || attention_mask.empty()
        || rope_cos.empty()
        || rope_sin.empty()
        || layer0_past_key.empty()
        || layer0_past_value.empty()
        || layer1_past_key.empty()
        || layer1_past_value.empty()
    ) {
        std::cerr
            << "创建一个或多个输入Mat失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "===== Runtime options =====\n"
        << "threads            : "
        << kThreads
        << '\n'
        << "packing layout     : "
        << (
            use_packing_layout
            ? "true"
            : "false"
        )
        << '\n'
        << "chain              : "
        << "Layer 0 out0 -> Layer 1 in0"
        << '\n';

    auto configure_network =
        [&use_packing_layout](
            ncnn::Net& network,
            const std::string& param_path,
            const std::string& model_path,
            const std::string& name)
        {
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
                    << "加载"
                    << name
                    << " param失败："
                    << param_path
                    << '\n';

                return false;
            }

            if (
                network.load_model(
                    model_path.c_str()
                ) != 0
            ) {
                std::cerr
                    << "加载"
                    << name
                    << " bin失败："
                    << model_path
                    << '\n';

                return false;
            }

            return true;
        };

    ncnn::Net layer0_network;
    ncnn::Net layer1_network;

    if (
        !configure_network(
            layer0_network,
            layer0_param_path,
            layer0_model_path,
            "Layer 0"
        )
        || !configure_network(
            layer1_network,
            layer1_param_path,
            layer1_model_path,
            "Layer 1"
        )
    ) {
        return EXIT_FAILURE;
    }

    auto bind_inputs = [](
        ncnn::Extractor& extractor,
        const ncnn::Mat& hidden,
        const ncnn::Mat& mask,
        const ncnn::Mat& cos,
        const ncnn::Mat& sin,
        const ncnn::Mat& past_key,
        const ncnn::Mat& past_value,
        const std::string& layer_name)
    {
        if (
            extractor.input("in0", hidden) != 0
            || extractor.input("in1", mask) != 0
            || extractor.input("in2", cos) != 0
            || extractor.input("in3", sin) != 0
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
                << layer_name
                << "写入输入失败。\n";

            return false;
        }

        return true;
    };

    auto extract_outputs = [](
        ncnn::Extractor& extractor,
        ncnn::Mat& output,
        ncnn::Mat& present_key,
        ncnn::Mat& present_value,
        const std::string& layer_name)
    {
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
                << layer_name
                << "提取输出失败："
                << "out0=" << output_status
                << "，out1=" << key_status
                << "，out2=" << value_status
                << '\n';

            return false;
        }

        return true;
    };

    /*
     * Layer 0运行。
     */
    ncnn::Extractor layer0_extractor =
        layer0_network.create_extractor();

    layer0_extractor.set_light_mode(false);

    if (
        !bind_inputs(
            layer0_extractor,
            layer0_hidden,
            attention_mask,
            rope_cos,
            rope_sin,
            layer0_past_key,
            layer0_past_value,
            "Layer 0"
        )
    ) {
        return EXIT_FAILURE;
    }

    ncnn::Mat layer0_output;
    ncnn::Mat layer0_present_key;
    ncnn::Mat layer0_present_value;

    if (
        !extract_outputs(
            layer0_extractor,
            layer0_output,
            layer0_present_key,
            layer0_present_value,
            "Layer 0"
        )
    ) {
        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Layer 0 outputs =====\n";

    print_shape(
        "layer0 out0",
        layer0_output
    );

    print_shape(
        "layer0 out1 present_key",
        layer0_present_key
    );

    print_shape(
        "layer0 out2 present_value",
        layer0_present_value
    );

    /*
     * 核心串联：
     * Layer 1的in0直接绑定Layer 0的ncnn::Mat out0。
     * 不重新读取layer1_hidden_states作为推理输入。
     */
    std::cout
        << "\n===== Direct ncnn::Mat handoff =====\n"
        << "Layer 1 in0 source: "
        << "Layer 0 out0 directly\n";

    ncnn::Extractor layer1_extractor =
        layer1_network.create_extractor();

    layer1_extractor.set_light_mode(false);

    if (
        !bind_inputs(
            layer1_extractor,
            layer0_output,
            attention_mask,
            rope_cos,
            rope_sin,
            layer1_past_key,
            layer1_past_value,
            "Layer 1"
        )
    ) {
        return EXIT_FAILURE;
    }

    ncnn::Mat layer1_output;
    ncnn::Mat layer1_present_key;
    ncnn::Mat layer1_present_value;

    if (
        !extract_outputs(
            layer1_extractor,
            layer1_output,
            layer1_present_key,
            layer1_present_value,
            "Layer 1"
        )
    ) {
        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Layer 1 outputs =====\n";

    print_shape(
        "layer1 out0",
        layer1_output
    );

    print_shape(
        "layer1 out1 present_key",
        layer1_present_key
    );

    print_shape(
        "layer1 out2 present_value",
        layer1_present_value
    );

    std::vector<float> actual_layer0_output;
    std::vector<float> actual_layer0_present_key;
    std::vector<float> actual_layer0_present_value;

    std::vector<float> actual_layer1_output;
    std::vector<float> actual_layer1_present_key;
    std::vector<float> actual_layer1_present_value;

    if (
        !unpack_mat(
            layer0_output,
            actual_layer0_output
        )
        || !unpack_mat(
            layer0_present_key,
            actual_layer0_present_key
        )
        || !unpack_mat(
            layer0_present_value,
            actual_layer0_present_value
        )
        || !unpack_mat(
            layer1_output,
            actual_layer1_output
        )
        || !unpack_mat(
            layer1_present_key,
            actual_layer1_present_key
        )
        || !unpack_mat(
            layer1_present_value,
            actual_layer1_present_value
        )
    ) {
        return EXIT_FAILURE;
    }

    Metrics reference_boundary_metrics;
    Metrics chain_boundary_metrics;

    Metrics layer0_output_metrics;
    Metrics layer0_key_metrics;
    Metrics layer0_value_metrics;

    Metrics layer1_output_metrics;
    Metrics layer1_key_metrics;
    Metrics layer1_value_metrics;

    if (
        !calculate_metrics(
            expected_layer0_output,
            expected_layer1_hidden,
            reference_boundary_metrics
        )
        || !calculate_metrics(
            actual_layer0_output,
            expected_layer1_hidden,
            chain_boundary_metrics
        )
        || !calculate_metrics(
            actual_layer0_output,
            expected_layer0_output,
            layer0_output_metrics
        )
        || !calculate_metrics(
            actual_layer0_present_key,
            expected_layer0_present_key,
            layer0_key_metrics
        )
        || !calculate_metrics(
            actual_layer0_present_value,
            expected_layer0_present_value,
            layer0_value_metrics
        )
        || !calculate_metrics(
            actual_layer1_output,
            expected_layer1_output,
            layer1_output_metrics
        )
        || !calculate_metrics(
            actual_layer1_present_key,
            expected_layer1_present_key,
            layer1_key_metrics
        )
        || !calculate_metrics(
            actual_layer1_present_value,
            expected_layer1_present_value,
            layer1_value_metrics
        )
    ) {
        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Reference boundary =====\n";

    print_metrics(
        "reference layer0 output vs layer1 hidden",
        reference_boundary_metrics
    );

    std::cout
        << "\n===== Direct chain boundary =====\n";

    print_metrics(
        "actual layer0 out0 vs expected layer1 hidden",
        chain_boundary_metrics
    );

    std::cout
        << "\n===== Layer 0 parity =====\n";

    print_metrics(
        "layer0 out0",
        layer0_output_metrics
    );

    print_metrics(
        "layer0 present_key",
        layer0_key_metrics
    );

    print_metrics(
        "layer0 present_value",
        layer0_value_metrics
    );

    std::cout
        << "\n===== Layer 1 parity =====\n";

    print_metrics(
        "layer1 out0",
        layer1_output_metrics
    );

    print_metrics(
        "layer1 present_key",
        layer1_key_metrics
    );

    print_metrics(
        "layer1 present_value",
        layer1_value_metrics
    );

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
                                static_cast<std::size_t>(
                                    head
                                )
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

                    const double error =
                        std::fabs(
                            static_cast<double>(
                                actual[index]
                            )
                            - static_cast<double>(
                                expected[index]
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

    const double layer0_key_prefix_error =
        cache_region_error(
            actual_layer0_present_key,
            expected_layer0_present_key,
            0,
            kPastLength
        );

    const double layer0_value_prefix_error =
        cache_region_error(
            actual_layer0_present_value,
            expected_layer0_present_value,
            0,
            kPastLength
        );

    const double layer1_key_prefix_error =
        cache_region_error(
            actual_layer1_present_key,
            expected_layer1_present_key,
            0,
            kPastLength
        );

    const double layer1_value_prefix_error =
        cache_region_error(
            actual_layer1_present_value,
            expected_layer1_present_value,
            0,
            kPastLength
        );

    std::cout
        << '\n'
        << std::scientific
        << std::setprecision(10)
        << "===== Cache history prefix checks =====\n"
        << "Layer 0 key prefix error   : "
        << layer0_key_prefix_error
        << '\n'
        << "Layer 0 value prefix error : "
        << layer0_value_prefix_error
        << '\n'
        << "Layer 1 key prefix error   : "
        << layer1_key_prefix_error
        << '\n'
        << "Layer 1 value prefix error : "
        << layer1_value_prefix_error
        << '\n';

    auto output_passed = [](
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

    const bool passed =
        reference_boundary_metrics
            .maximum_abs_error == 0.0
        && output_passed(
            chain_boundary_metrics
        )
        && output_passed(
            layer0_output_metrics
        )
        && cache_passed(
            layer0_key_metrics
        )
        && cache_passed(
            layer0_value_metrics
        )
        && output_passed(
            layer1_output_metrics
        )
        && cache_passed(
            layer1_key_metrics
        )
        && cache_passed(
            layer1_value_metrics
        )
        && layer0_key_prefix_error == 0.0
        && layer0_value_prefix_error == 0.0
        && layer1_key_prefix_error == 0.0
        && layer1_value_prefix_error == 0.0;

    if (!passed) {
        std::cerr
            << "\n❌ Decoder Layer 0 → Layer 1 "
            << "两层串联数值验证失败。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "\n✅ Decoder Layer 0 → Layer 1 "
        << "ncnn::Mat直接串联数值验证成功。\n";

    return EXIT_SUCCESS;
}
