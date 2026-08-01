#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <net.h>

namespace {

constexpr int kSequenceLength = 313;
constexpr int kHiddenSize = 1024;
constexpr int kHeadDim = 128;
constexpr int kRopeComponents = 4;

constexpr std::size_t kHiddenCount =
    static_cast<std::size_t>(kSequenceLength) *
    static_cast<std::size_t>(kHiddenSize);

constexpr std::size_t kMaskCount =
    static_cast<std::size_t>(kSequenceLength) *
    static_cast<std::size_t>(kSequenceLength);

constexpr std::size_t kRopeCount =
    static_cast<std::size_t>(kRopeComponents) *
    static_cast<std::size_t>(kSequenceLength) *
    static_cast<std::size_t>(kHeadDim);


struct Checkpoint {
    const char* stage;
    const char* blob;
    const char* reference_file;
};


struct Metrics {
    float maximum_absolute_error = 0.0f;
    double mean_absolute_error = 0.0;
    double root_mean_square_error = 0.0;
    double cosine_similarity = 0.0;
    std::size_t maximum_error_index = 0;
};


bool load_all_f32(
    const std::string& path,
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

    const std::streamsize bytes = file.tellg();

    if (
        bytes <= 0 ||
        bytes % static_cast<std::streamsize>(
            sizeof(float)
        ) != 0
    ) {
        std::cerr
            << "非法FP32文件大小："
            << path
            << "，bytes="
            << bytes
            << '\n';
        return false;
    }

    file.seekg(0, std::ios::beg);

    values.resize(
        static_cast<std::size_t>(
            bytes / sizeof(float)
        )
    );

    if (!file.read(
            reinterpret_cast<char*>(
                values.data()
            ),
            bytes
        )) {
        std::cerr
            << "读取文件失败："
            << path
            << '\n';
        return false;
    }

    return true;
}


bool load_exact_f32(
    const std::string& path,
    const std::size_t expected_count,
    std::vector<float>& values)
{
    if (!load_all_f32(path, values)) {
        return false;
    }

    if (values.size() != expected_count) {
        std::cerr
            << "元素数错误："
            << path
            << "\n实际="
            << values.size()
            << "，预期="
            << expected_count
            << '\n';
        return false;
    }

    return true;
}


std::size_t logical_count(
    const ncnn::Mat& value)
{
    return
        static_cast<std::size_t>(value.w) *
        static_cast<std::size_t>(value.h) *
        static_cast<std::size_t>(value.d) *
        static_cast<std::size_t>(value.c) *
        static_cast<std::size_t>(
            value.elempack
        );
}


void print_shape(
    const ncnn::Mat& value)
{
    std::cout
        << "dims=" << value.dims
        << ",w=" << value.w
        << ",h=" << value.h
        << ",d=" << value.d
        << ",c=" << value.c
        << ",pack=" << value.elempack;
}


Metrics calculate_metrics(
    const float* actual,
    const std::vector<float>& expected)
{
    Metrics metrics;

    double absolute_error_sum = 0.0;
    double squared_error_sum = 0.0;

    double dot_product = 0.0;
    double actual_norm_square = 0.0;
    double expected_norm_square = 0.0;

    for (
        std::size_t index = 0;
        index < expected.size();
        ++index
    ) {
        const double actual_value =
            actual[index];

        const double expected_value =
            expected[index];

        const double error =
            std::fabs(
                actual_value - expected_value
            );

        if (
            error >
            metrics.maximum_absolute_error
        ) {
            metrics.maximum_absolute_error =
                static_cast<float>(error);

            metrics.maximum_error_index =
                index;
        }

        absolute_error_sum += error;
        squared_error_sum += error * error;

        dot_product +=
            actual_value * expected_value;

        actual_norm_square +=
            actual_value * actual_value;

        expected_norm_square +=
            expected_value * expected_value;
    }

    const double count =
        static_cast<double>(
            expected.size()
        );

    metrics.mean_absolute_error =
        absolute_error_sum / count;

    metrics.root_mean_square_error =
        std::sqrt(
            squared_error_sum / count
        );

    const double denominator =
        std::sqrt(actual_norm_square) *
        std::sqrt(expected_norm_square);

    metrics.cosine_similarity =
        denominator > 0.0
            ? dot_product / denominator
            : 0.0;

    return metrics;
}


bool compare_blob(
    ncnn::Extractor& extractor,
    const Checkpoint& checkpoint,
    const std::string& reference_dir,
    Metrics& metrics)
{
    ncnn::Mat output;

    const int extract_result =
        extractor.extract(
            checkpoint.blob,
            output
        );

    if (extract_result != 0) {
        std::cerr
            << "提取失败：stage="
            << checkpoint.stage
            << "，blob="
            << checkpoint.blob
            << "，返回值="
            << extract_result
            << '\n';
        return false;
    }

    if (output.empty()) {
        std::cerr
            << "提取结果为空："
            << checkpoint.stage
            << '\n';
        return false;
    }

    if (output.elempack != 1) {
        std::cerr
            << "意外的elempack："
            << checkpoint.stage
            << "，elempack="
            << output.elempack
            << '\n';
        return false;
    }

    std::vector<float> expected;

    const std::string reference_path =
        reference_dir
        + "/"
        + checkpoint.reference_file;

    if (!load_all_f32(
            reference_path,
            expected
        )) {
        return false;
    }

    const std::size_t actual_count =
        logical_count(output);

    if (actual_count != expected.size()) {
        std::cerr
            << "元素数不一致："
            << checkpoint.stage
            << "\nblob=";

        print_shape(output);

        std::cerr
            << "\n实际="
            << actual_count
            << "，参考="
            << expected.size()
            << '\n';

        return false;
    }

    ncnn::Mat flat = output.reshape(
        static_cast<int>(actual_count)
    );

    if (flat.empty()) {
        std::cerr
            << "flatten失败："
            << checkpoint.stage
            << '\n';
        return false;
    }

    const float* actual = flat;

    for (
        std::size_t index = 0;
        index < actual_count;
        ++index
    ) {
        if (!std::isfinite(actual[index])) {
            std::cerr
                << "发现非有限值："
                << checkpoint.stage
                << "，index="
                << index
                << '\n';
            return false;
        }
    }

    metrics = calculate_metrics(
        actual,
        expected
    );

    std::cout
        << std::left
        << std::setw(34)
        << checkpoint.stage
        << " blob="
        << std::setw(5)
        << checkpoint.blob
        << " ";

    print_shape(output);

    std::cout
        << std::right
        << std::scientific
        << std::setprecision(6)
        << "  max="
        << metrics.maximum_absolute_error
        << "  mean="
        << metrics.mean_absolute_error
        << "  rmse="
        << metrics.root_mean_square_error
        << "  cos="
        << std::fixed
        << std::setprecision(10)
        << metrics.cosine_similarity
        << "  max_index="
        << metrics.maximum_error_index
        << '\n';

    return true;
}

} // namespace


int main(int argc, char** argv)
{
    if (argc != 8) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <param>"
            << " <bin>"
            << " <hidden_states>"
            << " <attention_mask>"
            << " <rope_cos>"
            << " <rope_sin>"
            << " <debug_ref_dir>\n";

        return EXIT_FAILURE;
    }

    const std::string param_path = argv[1];
    const std::string model_path = argv[2];
    const std::string hidden_path = argv[3];
    const std::string mask_path = argv[4];
    const std::string cos_path = argv[5];
    const std::string sin_path = argv[6];
    const std::string reference_dir = argv[7];

    std::vector<float> hidden_values;
    std::vector<float> mask_values;
    std::vector<float> cos_values;
    std::vector<float> sin_values;

    if (
        !load_exact_f32(
            hidden_path,
            kHiddenCount,
            hidden_values
        ) ||
        !load_exact_f32(
            mask_path,
            kMaskCount,
            mask_values
        ) ||
        !load_exact_f32(
            cos_path,
            kRopeCount,
            cos_values
        ) ||
        !load_exact_f32(
            sin_path,
            kRopeCount,
            sin_values
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

    ncnn::Mat mask_flat(
        static_cast<int>(kMaskCount),
        static_cast<void*>(
            mask_values.data()
        )
    );

    ncnn::Mat cos_flat(
        static_cast<int>(kRopeCount),
        static_cast<void*>(
            cos_values.data()
        )
    );

    ncnn::Mat sin_flat(
        static_cast<int>(kRopeCount),
        static_cast<void*>(
            sin_values.data()
        )
    );

    ncnn::Mat hidden_states =
        hidden_flat.reshape(
            kHiddenSize,
            kSequenceLength
        ).clone();

    ncnn::Mat attention_mask =
        mask_flat.reshape(
            kSequenceLength,
            kSequenceLength,
            1
        ).clone();

    ncnn::Mat rope_cos =
        cos_flat.reshape(
            kHeadDim,
            kSequenceLength,
            kRopeComponents
        ).clone();

    ncnn::Mat rope_sin =
        sin_flat.reshape(
            kHeadDim,
            kSequenceLength,
            kRopeComponents
        ).clone();

    ncnn::Net network;

    network.opt.use_vulkan_compute = false;
    network.opt.use_packing_layout = false;
    network.opt.num_threads = 1;

    if (
        network.load_param(
            param_path.c_str()
        ) != 0
    ) {
        std::cerr << "加载param失败。\n";
        return EXIT_FAILURE;
    }

    if (
        network.load_model(
            model_path.c_str()
        ) != 0
    ) {
        std::cerr << "加载bin失败。\n";
        return EXIT_FAILURE;
    }

    ncnn::Extractor extractor =
        network.create_extractor();

    extractor.set_light_mode(false);

    if (
        extractor.input(
            "in0",
            hidden_states
        ) != 0 ||
        extractor.input(
            "in1",
            attention_mask
        ) != 0 ||
        extractor.input(
            "in2",
            rope_cos
        ) != 0 ||
        extractor.input(
            "in3",
            rope_sin
        ) != 0
    ) {
        std::cerr << "输入网络失败。\n";
        return EXIT_FAILURE;
    }

    const Checkpoint checkpoints[] = {
        {
            "input RMSNorm",
            "6",
            "input_layernorm_output_f32.bin",
        },
        {
            "Q projection",
            "10",
            "q_projection_output_f32.bin",
        },
        {
            "K projection",
            "13",
            "k_projection_output_f32.bin",
        },
        {
            "V projection",
            "16",
            "v_projection_output_f32.bin",
        },
        {
            "Q after RoPE",
            "51",
            "query_layernorm_input_f32.bin",
        },
        {
            "Q head RMSNorm",
            "52",
            "query_layernorm_output_f32.bin",
        },
        {
            "K after RoPE",
            "53",
            "key_layernorm_input_f32.bin",
        },
        {
            "K head RMSNorm",
            "54",
            "key_layernorm_output_f32.bin",
        },
        {
            "SDPA / O projection input",
            "63",
            "o_projection_input_f32.bin",
        },
        {
            "O projection",
            "64",
            "o_projection_output_f32.bin",
        },
        {
            "attention residual",
            "65",
            "post_attention_layernorm_input_f32.bin",
        },
        {
            "post-attention RMSNorm",
            "68",
            "post_attention_layernorm_output_f32.bin",
        },
        {
            "MLP gate projection",
            "71",
            "mlp_gate_output_f32.bin",
        },
        {
            "MLP up projection",
            "73",
            "mlp_up_output_f32.bin",
        },
        {
            "SwiGLU product",
            "74",
            "mlp_down_input_f32.bin",
        },
        {
            "MLP down projection",
            "75",
            "mlp_down_output_f32.bin",
        },
        {
            "Layer output",
            "out0",
            "layer_output_f32.bin",
        },
    };

    std::cout
        << "===== Decoder Layer 0 "
        << "intermediate parity =====\n";

    bool all_extracted = true;
    bool first_suspicious_found = false;

    for (const Checkpoint& checkpoint :
         checkpoints) {
        Metrics metrics;

        if (!compare_blob(
                extractor,
                checkpoint,
                reference_dir,
                metrics
            )) {
            all_extracted = false;
            continue;
        }

        const bool suspicious =
            metrics.maximum_absolute_error >
                1e-4f ||
            metrics.mean_absolute_error >
                1e-6 ||
            metrics.cosine_similarity <
                0.9999999;

        if (
            suspicious &&
            !first_suspicious_found
        ) {
            std::cout
                << ">>> 第一个明显偏差阶段："
                << checkpoint.stage
                << "，blob="
                << checkpoint.blob
                << '\n';

            first_suspicious_found = true;
        }
    }

    if (!all_extracted) {
        return EXIT_FAILURE;
    }

    std::cout
        << "✅ 所有中间blob均已完成比较。\n";

    return EXIT_SUCCESS;
}
