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
constexpr int kMaskLength = 316;

constexpr int kRopeComponents = 4;
constexpr int kHeadDim = 128;

constexpr int kKeyValueHeads = 8;
constexpr int kPastLength = 315;
constexpr int kPresentLength = 316;

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


bool save_float_binary(
    const std::string& path,
    const std::vector<float>& values)
{
    std::ofstream file(
        path,
        std::ios::binary
    );

    if (!file.is_open()) {
        std::cerr
            << "无法创建中间张量文件："
            << path
            << '\n';

        return false;
    }

    const std::streamsize byte_count =
        static_cast<std::streamsize>(
            values.size() * sizeof(float)
        );

    file.write(
        reinterpret_cast<const char*>(
            values.data()
        ),
        byte_count
    );

    if (!file.good()) {
        std::cerr
            << "写入中间张量失败："
            << path
            << '\n';

        return false;
    }

    return true;
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


double cache_region_max_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const int token_begin,
    const int token_end)
{
    double maximum_error = 0.0;

    for (
        int head = 0;
        head < kKeyValueHeads;
        ++head
    ) {
        for (
            int token = token_begin;
            token < token_end;
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

                maximum_error = std::max(
                    maximum_error,
                    std::abs(
                        static_cast<double>(
                            actual[index]
                        )
                        - static_cast<double>(
                            expected[index]
                        )
                    )
                );
            }
        }
    }

    return maximum_error;
}


bool passes_threshold(
    const Metrics& metrics,
    const double maximum_tolerance,
    const double mean_tolerance,
    const double minimum_cosine)
{
    return
        metrics.maximum_abs_error
            <= maximum_tolerance
        && metrics.mean_abs_error
            <= mean_tolerance
        && metrics.cosine_similarity
            >= minimum_cosine;
}

} // namespace


int main(
    int argc,
    char** argv)
{
    if (argc != 13) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <ncnn.param>"
            << " <ncnn.bin>"
            << " <hidden_f32.bin>"
            << " <mask_f32.bin>"
            << " <rope_cos_f32.bin>"
            << " <rope_sin_f32.bin>"
            << " <past_key_f32.bin>"
            << " <past_value_f32.bin>"
            << " <expected_output_f32.bin>"
            << " <expected_present_key_f32.bin>"
            << " <expected_present_value_f32.bin>"
            << " <packing:0|1>\n";

        return EXIT_FAILURE;
    }

    const std::string param_path = argv[1];
    const std::string model_path = argv[2];

    const bool use_packing_layout =
        std::string(argv[12]) == "1";

    std::vector<float> hidden_values;
    std::vector<float> mask_values;
    std::vector<float> rope_cos_values;
    std::vector<float> rope_sin_values;
    std::vector<float> past_key_values;
    std::vector<float> past_value_values;

    std::vector<float> expected_output;
    std::vector<float> expected_present_key;
    std::vector<float> expected_present_value;

    if (
        !load_exact_binary(
            argv[3],
            kHiddenCount,
            hidden_values
        )
        || !load_exact_binary(
            argv[4],
            kMaskCount,
            mask_values
        )
        || !load_exact_binary(
            argv[5],
            kRopeCount,
            rope_cos_values
        )
        || !load_exact_binary(
            argv[6],
            kRopeCount,
            rope_sin_values
        )
        || !load_exact_binary(
            argv[7],
            kPastCacheCount,
            past_key_values
        )
        || !load_exact_binary(
            argv[8],
            kPastCacheCount,
            past_value_values
        )
        || !load_exact_binary(
            argv[9],
            kHiddenCount,
            expected_output
        )
        || !load_exact_binary(
            argv[10],
            kPresentCacheCount,
            expected_present_key
        )
        || !load_exact_binary(
            argv[11],
            kPresentCacheCount,
            expected_present_value
        )
    ) {
        return EXIT_FAILURE;
    }

    ncnn::Mat hidden_flat(
        static_cast<int>(kHiddenCount),
        static_cast<void*>(
            hidden_values.data()
        )
    );

    // PyTorch [1, 1, 1024]，batch_index=0。
    // 去除batch轴后保留 [1, 1024]，对应ncnn dims=2。
    ncnn::Mat hidden_states =
        hidden_flat.reshape(
            kHiddenSize,
            1
        ).clone();

    ncnn::Mat mask_flat(
        static_cast<int>(kMaskCount),
        static_cast<void*>(
            mask_values.data()
        )
    );

    // PyTorch [1, 1, 1, 314]，batch_index=0。
    // 去除batch轴后保留 [1, 1, 314]，对应ncnn dims=3。
    ncnn::Mat attention_mask =
        mask_flat.reshape(
            kMaskLength,
            1,
            1
        ).clone();

    ncnn::Mat rope_cos_flat(
        static_cast<int>(kRopeCount),
        static_cast<void*>(
            rope_cos_values.data()
        )
    );

    ncnn::Mat rope_cos =
        rope_cos_flat.reshape(
            kHeadDim,
            1,
            kRopeComponents
        ).clone();

    ncnn::Mat rope_sin_flat(
        static_cast<int>(kRopeCount),
        static_cast<void*>(
            rope_sin_values.data()
        )
    );

    ncnn::Mat rope_sin =
        rope_sin_flat.reshape(
            kHeadDim,
            1,
            kRopeComponents
        ).clone();

    ncnn::Mat past_key_flat(
        static_cast<int>(kPastCacheCount),
        static_cast<void*>(
            past_key_values.data()
        )
    );

    ncnn::Mat past_key =
        past_key_flat.reshape(
            kHeadDim,
            kPastLength,
            kKeyValueHeads
        ).clone();

    ncnn::Mat past_value_flat(
        static_cast<int>(kPastCacheCount),
        static_cast<void*>(
            past_value_values.data()
        )
    );

    ncnn::Mat past_value =
        past_value_flat.reshape(
            kHeadDim,
            kPastLength,
            kKeyValueHeads
        ).clone();

    if (
        hidden_states.empty()
        || attention_mask.empty()
        || rope_cos.empty()
        || rope_sin.empty()
        || past_key.empty()
        || past_value.empty()
    ) {
        std::cerr
            << "一个或多个输入Mat创建失败。\n";

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
        << '\n';

    std::cout
        << "\n===== Input shapes =====\n";

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

    print_shape(
        "in4 past_key",
        past_key
    );

    print_shape(
        "in5 past_value",
        past_value
    );

    ncnn::Net network;

    network.opt.use_vulkan_compute = false;
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
            << "加载ncnn param失败："
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
            << "加载ncnn bin失败："
            << model_path
            << '\n';

        return EXIT_FAILURE;
    }

    /*
     * packing布局诊断：
     * 每个blob都使用全新的Extractor。
     *
     * 这样即使某次extract失败，也不会影响后续独立检查；
     * 同时可以定位第一个无法执行的packed-layout区间。
     */
    const char* dump_directory =
        std::getenv("NCNN_DUMP_DIR");

    const char* probe_flag =
        std::getenv("NCNN_PROBE");

    const bool enable_probe =
        dump_directory != nullptr
        || (
            probe_flag != nullptr
            && std::string(probe_flag) == "1"
        );

    if (enable_probe) {
        struct ProbePoint {
            const char* blob_name;
            const char* description;
        };

        const ProbePoint checkpoints[] = {
            {"8", "input RMSNorm"},
            {"12", "Q projection"},
            {"13", "K projection"},
            {"14", "V projection"},

            {"16", "Q reshape"},
            {"18", "K reshape"},
            {"20", "V reshape"},

            {"33", "merged RoPE cosine"},
            {"48", "merged RoPE sine"},

            {"51", "Q after RotaryEmbed"},
            {"52", "Q head RMSNorm"},
            {"53", "K after RotaryEmbed"},
            {"54", "K head RMSNorm"},

            {"55", "present key before output split"},
            {"out1", "present key output"},

            {"58", "present value before output split"},
            {"out2", "present value output"},

            {"63", "repeated key"},
            {"66", "repeated value"},

            {"67", "attention scores"},
            {"68", "scaled attention scores"},
            {"69", "masked attention scores"},
            {"70", "attention probabilities"},
            {"71", "attention context"},

            {"73", "O projection input"},
            {"74", "O projection"},
            {"75", "attention residual"},

            {"78", "post-attention RMSNorm"},
            {"81", "MLP gate projection"},
            {"82", "MLP up projection"},
            {"84", "SwiGLU product"},
            {"85", "MLP down projection"},

            {"out0", "layer output"},
        };

        std::cout
            << "\n===== Packing checkpoint probe =====\n";

        for (const ProbePoint& point : checkpoints) {
            ncnn::Extractor probe_extractor =
                network.create_extractor();

            probe_extractor.set_light_mode(false);

            const int input0_status =
                probe_extractor.input(
                    "in0",
                    hidden_states
                );

            const int input1_status =
                probe_extractor.input(
                    "in1",
                    attention_mask
                );

            const int input2_status =
                probe_extractor.input(
                    "in2",
                    rope_cos
                );

            const int input3_status =
                probe_extractor.input(
                    "in3",
                    rope_sin
                );

            const int input4_status =
                probe_extractor.input(
                    "in4",
                    past_key
                );

            const int input5_status =
                probe_extractor.input(
                    "in5",
                    past_value
                );

            if (
                input0_status != 0
                || input1_status != 0
                || input2_status != 0
                || input3_status != 0
                || input4_status != 0
                || input5_status != 0
            ) {
                std::cout
                    << "❌ 输入写入失败，停止探针。\n";

                break;
            }

            ncnn::Mat probe_output;

            const int status =
                probe_extractor.extract(
                    point.blob_name,
                    probe_output
                );

            std::cout
                << "blob="
                << std::setw(5)
                << point.blob_name
                << "  "
                << std::left
                << std::setw(34)
                << point.description
                << std::right
                << " status="
                << status;

            if (status == 0) {
                std::cout
                    << "  dims="
                    << probe_output.dims
                    << ",w="
                    << probe_output.w
                    << ",h="
                    << probe_output.h
                    << ",c="
                    << probe_output.c
                    << ",pack="
                    << probe_output.elempack;
            }

            std::cout << '\n';

            if (
                status == 0
                && dump_directory != nullptr
            ) {
                std::vector<float> logical_values;

                if (!unpack_mat(
                        probe_output,
                        logical_values
                    )) {
                    std::cerr
                        << "无法展开探针blob："
                        << point.blob_name
                        << '\n';

                    return EXIT_FAILURE;
                }

                const std::string dump_path =
                    std::string(dump_directory)
                    + "/blob_"
                    + point.blob_name
                    + ".f32.bin";

                if (!save_float_binary(
                        dump_path,
                        logical_values
                    )) {
                    return EXIT_FAILURE;
                }
            }

            if (status != 0) {
                std::cout
                    << "❌ 第一个失败检查点："
                    << point.description
                    << "，blob="
                    << point.blob_name
                    << '\n';

                break;
            }
        }

        std::cout
            << "===== End packing checkpoint probe =====\n";
    }

    ncnn::Extractor extractor =
        network.create_extractor();

    extractor.set_light_mode(false);

    if (
        extractor.input(
            "in0",
            hidden_states
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
            << "向Extractor写入输入失败。\n";

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
            << "提取输出失败："
            << "out0=" << output_status
            << "，out1=" << key_status
            << "，out2=" << value_status
            << '\n';

        return EXIT_FAILURE;
    }

    std::cout
        << "\n===== Output shapes =====\n";

    print_shape("out0 layer_output", output);
    print_shape("out1 present_key", present_key);
    print_shape("out2 present_value", present_value);

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

    std::cout
        << "\n===== Numerical parity =====\n";

    print_metrics(
        "out0 layer_output",
        output_metrics
    );

    print_metrics(
        "out1 present_key",
        key_metrics
    );

    print_metrics(
        "out2 present_value",
        value_metrics
    );

    const double key_prefix_error =
        cache_region_max_error(
            actual_present_key,
            expected_present_key,
            0,
            kPastLength
        );

    const double key_new_token_error =
        cache_region_max_error(
            actual_present_key,
            expected_present_key,
            kPastLength,
            kPresentLength
        );

    const double value_prefix_error =
        cache_region_max_error(
            actual_present_value,
            expected_present_value,
            0,
            kPastLength
        );

    const double value_new_token_error =
        cache_region_max_error(
            actual_present_value,
            expected_present_value,
            kPastLength,
            kPresentLength
        );

    std::cout
        << '\n'
        << std::scientific
        << std::setprecision(10)
        << "===== Cache region checks =====\n"
        << "Key history prefix max error : "
        << key_prefix_error
        << '\n'
        << "Key appended token max error : "
        << key_new_token_error
        << '\n'
        << "Value history prefix max error: "
        << value_prefix_error
        << '\n'
        << "Value appended token max error: "
        << value_new_token_error
        << '\n';

    const bool output_passed =
        passes_threshold(
            output_metrics,
            2.0e-3,
            1.0e-5,
            0.99999999
        );

    const bool key_passed =
        passes_threshold(
            key_metrics,
            2.0e-4,
            1.0e-6,
            0.99999999
        );

    const bool value_passed =
        passes_threshold(
            value_metrics,
            2.0e-4,
            1.0e-6,
            0.99999999
        );

    if (
        !output_passed
        || !key_passed
        || !value_passed
    ) {
        std::cerr
            << "\n❌ Decoder Layer 0 Decode Step 3 "
            << "ncnn C++数值误差超限。\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "\n✅ Decoder Layer 0 Decode Step 3 "
        << "PNNX → ncnn C++三输出数值对齐成功。\n";

    return EXIT_SUCCESS;
}
