#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <net.h>

namespace {

constexpr int kLayerCount = 24;
constexpr int kRuntimeThreads = 9;
constexpr int kHiddenSize = 1024;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kRopeComponents = 4;
constexpr int kPrefillLength = 313;
constexpr int kDecodeSteps = 10;
constexpr int kVocabSize = 120818;
constexpr int kEosToken = 120007;

constexpr std::size_t kPrefillHiddenCount =
    static_cast<std::size_t>(kPrefillLength) * kHiddenSize;
constexpr std::size_t kPrefillMaskCount =
    static_cast<std::size_t>(kPrefillLength) * kPrefillLength;
constexpr std::size_t kPrefillRopeCount =
    static_cast<std::size_t>(kRopeComponents) * kPrefillLength * kHeadDim;
constexpr std::size_t kPrefillCacheCount =
    static_cast<std::size_t>(kKvHeads) * kPrefillLength * kHeadDim;

constexpr std::array<int, kDecodeSteps + 1> kExpectedTokens = {
    93892, 5112, 206, 1717, 21, 185,
    18009, 15613, 16678, 21836, 120007,
};

constexpr const char* kExpectedText = "HELLO 2026\nNCNN CPU TEST";

struct Metrics {
    double maximum_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double cosine_similarity = 0.0;
};

template <typename T>
bool load_exact_binary(
    const std::string& path,
    std::size_t expected_count,
    std::vector<T>& values)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Unable to open " << path << '\n';
        return false;
    }
    const std::streamsize expected_bytes =
        static_cast<std::streamsize>(expected_count * sizeof(T));
    if (file.tellg() != expected_bytes) {
        std::cerr << "Unexpected file size for " << path
                  << ": actual=" << file.tellg()
                  << ", expected=" << expected_bytes << '\n';
        return false;
    }
    file.seekg(0, std::ios::beg);
    values.resize(expected_count);
    return static_cast<bool>(file.read(
        reinterpret_cast<char*>(values.data()), expected_bytes));
}

std::size_t logical_count(const ncnn::Mat& value)
{
    return static_cast<std::size_t>(value.w)
        * static_cast<std::size_t>(value.h)
        * static_cast<std::size_t>(value.d)
        * static_cast<std::size_t>(value.c)
        * static_cast<std::size_t>(value.elempack);
}

bool unpack_mat(const ncnn::Mat& value, std::vector<float>& result)
{
    if (value.empty()
        || value.elemsize != sizeof(float)
            * static_cast<std::size_t>(value.elempack)) {
        return false;
    }
    result.assign(logical_count(value), 0.0f);
    const int pack = value.elempack;
    if (value.dims == 1) {
        const float* source = value;
        std::copy(source, source + result.size(), result.begin());
        return true;
    }
    if (value.dims == 2) {
        for (int y = 0; y < value.h; ++y) {
            const float* source = value.row(y);
            for (int x = 0; x < value.w; ++x) {
                for (int p = 0; p < pack; ++p) {
                    const std::size_t destination =
                        static_cast<std::size_t>(y * pack + p)
                        * static_cast<std::size_t>(value.w)
                        + static_cast<std::size_t>(x);
                    result[destination] = source[x * pack + p];
                }
            }
        }
        return true;
    }
    if (value.dims == 3) {
        for (int q = 0; q < value.c; ++q) {
            const float* source = value.channel(q);
            for (int y = 0; y < value.h; ++y) {
                for (int x = 0; x < value.w; ++x) {
                    for (int p = 0; p < pack; ++p) {
                        const std::size_t destination =
                            (static_cast<std::size_t>(q * pack + p)
                                * static_cast<std::size_t>(value.h)
                                + static_cast<std::size_t>(y))
                            * static_cast<std::size_t>(value.w)
                            + static_cast<std::size_t>(x);
                        const std::size_t source_index =
                            (static_cast<std::size_t>(y)
                                * static_cast<std::size_t>(value.w)
                                + static_cast<std::size_t>(x))
                            * static_cast<std::size_t>(pack)
                            + static_cast<std::size_t>(p);
                        result[destination] = source[source_index];
                    }
                }
            }
        }
        return true;
    }
    return false;
}

bool calculate_metrics(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    Metrics& metrics)
{
    if (actual.size() != expected.size() || actual.empty()) {
        return false;
    }
    long double absolute_sum = 0.0;
    long double dot = 0.0;
    long double actual_norm = 0.0;
    long double expected_norm = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double a = actual[index];
        const double e = expected[index];
        if (!std::isfinite(a) || !std::isfinite(e)) {
            return false;
        }
        const double difference = std::abs(a - e);
        metrics.maximum_abs_error = std::max(
            metrics.maximum_abs_error, difference);
        absolute_sum += difference;
        dot += static_cast<long double>(a) * e;
        actual_norm += static_cast<long double>(a) * a;
        expected_norm += static_cast<long double>(e) * e;
    }
    metrics.mean_abs_error = static_cast<double>(
        absolute_sum / static_cast<long double>(actual.size()));
    const long double denominator = std::sqrt(actual_norm * expected_norm);
    metrics.cosine_similarity = denominator == 0.0
        ? (actual == expected ? 1.0 : 0.0)
        : static_cast<double>(dot / denominator);
    metrics.cosine_similarity = std::clamp(
        metrics.cosine_similarity, -1.0, 1.0);
    return true;
}

bool hidden_passed(const Metrics& metrics)
{
    return metrics.maximum_abs_error <= 2.0e-3
        && metrics.mean_abs_error <= 2.0e-4
        && metrics.cosine_similarity >= 0.999999;
}

bool cache_passed(const Metrics& metrics)
{
    return metrics.maximum_abs_error <= 2.0e-4
        && metrics.mean_abs_error <= 2.0e-6
        && metrics.cosine_similarity >= 0.999999;
}

bool prefill_hidden_passed(const Metrics& metrics)
{
    return metrics.maximum_abs_error <= 1.0e-3
        && metrics.mean_abs_error <= 1.0e-5
        && metrics.cosine_similarity >= 0.999999;
}

bool prefill_cache_passed(const Metrics& metrics)
{
    return metrics.maximum_abs_error <= 2.0e-4
        && metrics.mean_abs_error <= 2.0e-6
        && metrics.cosine_similarity >= 0.999999;
}

ncnn::Mat make_hidden(std::vector<float>& values)
{
    ncnn::Mat flat(static_cast<int>(values.size()), values.data());
    return flat.reshape(kHiddenSize, 1).clone();
}

ncnn::Mat make_prefill_hidden(std::vector<float>& values)
{
    ncnn::Mat flat(static_cast<int>(values.size()), values.data());
    return flat.reshape(kHiddenSize, kPrefillLength).clone();
}

ncnn::Mat make_prefill_mask(std::vector<float>& values)
{
    ncnn::Mat flat(static_cast<int>(values.size()), values.data());
    return flat.reshape(kPrefillLength, kPrefillLength, 1).clone();
}

ncnn::Mat make_prefill_rope(std::vector<float>& values)
{
    ncnn::Mat flat(static_cast<int>(values.size()), values.data());
    return flat.reshape(kHeadDim, kPrefillLength, kRopeComponents).clone();
}

ncnn::Mat make_mask(std::vector<float>& values, int length)
{
    ncnn::Mat flat(static_cast<int>(values.size()), values.data());
    return flat.reshape(length, 1, 1).clone();
}

ncnn::Mat make_rope(std::vector<float>& values)
{
    ncnn::Mat flat(static_cast<int>(values.size()), values.data());
    return flat.reshape(kHeadDim, 1, kRopeComponents).clone();
}

std::string decode_name(int layer, int step)
{
    std::ostringstream name;
    name << "decoder_layer" << layer << "_decode";
    if (step > 1) {
        name << "_step" << step;
    }
    return name.str();
}

std::string reference_directory(
    const std::string& project_root,
    int layer,
    int step)
{
    return project_root + "/artifacts/" + decode_name(layer, step)
        + "/reference";
}

std::string prefill_reference_directory(
    const std::string& project_root,
    int layer)
{
    const std::string name = "decoder_layer" + std::to_string(layer)
        + "_prefill_kv";
    return project_root + "/artifacts/" + name + "/reference";
}

std::string prefill_tail_directory(const std::string& project_root)
{
    return project_root + "/artifacts/prefill_tail/reference";
}

std::string tail_directory(const std::string& project_root, int step)
{
    if (step == 1) {
        return project_root + "/artifacts/decode_tail/reference";
    }
    return project_root + "/artifacts/decode_tail_step"
        + std::to_string(step) + "/reference";
}

bool configure_and_load(
    ncnn::Net& network,
    const std::string& param_path,
    const std::string& model_path,
    bool use_packing_layout)
{
    network.opt.use_vulkan_compute = false;
    network.opt.use_packing_layout = use_packing_layout;
    network.opt.num_threads = kRuntimeThreads;
    return network.load_param(param_path.c_str()) == 0
        && network.load_model(model_path.c_str()) == 0;
}

bool run_decoder_layer(
    ncnn::Net& network,
    const ncnn::Mat& hidden,
    const ncnn::Mat& mask,
    const ncnn::Mat& rope_cos,
    const ncnn::Mat& rope_sin,
    const ncnn::Mat& past_key,
    const ncnn::Mat& past_value,
    ncnn::Mat& output,
    ncnn::Mat& present_key,
    ncnn::Mat& present_value)
{
    ncnn::Extractor extractor = network.create_extractor();
    extractor.set_light_mode(false);
    if (extractor.input("in0", hidden) != 0
        || extractor.input("in1", mask) != 0
        || extractor.input("in2", rope_cos) != 0
        || extractor.input("in3", rope_sin) != 0
        || extractor.input("in4", past_key) != 0
        || extractor.input("in5", past_value) != 0) {
        return false;
    }
    ncnn::Mat raw_output;
    ncnn::Mat raw_key;
    ncnn::Mat raw_value;
    if (extractor.extract("out0", raw_output) != 0
        || extractor.extract("out1", raw_key) != 0
        || extractor.extract("out2", raw_value) != 0) {
        return false;
    }
    output = raw_output.clone();
    present_key = raw_key.clone();
    present_value = raw_value.clone();
    return !output.empty() && !present_key.empty() && !present_value.empty();
}

bool run_prefill_layer(
    ncnn::Net& network,
    const ncnn::Mat& hidden,
    const ncnn::Mat& mask,
    const ncnn::Mat& rope_cos,
    const ncnn::Mat& rope_sin,
    ncnn::Mat& output,
    ncnn::Mat& present_key,
    ncnn::Mat& present_value)
{
    ncnn::Extractor extractor = network.create_extractor();
    extractor.set_light_mode(false);
    if (extractor.input("in0", hidden) != 0
        || extractor.input("in1", mask) != 0
        || extractor.input("in2", rope_cos) != 0
        || extractor.input("in3", rope_sin) != 0) {
        return false;
    }
    ncnn::Mat raw_output;
    ncnn::Mat raw_key;
    ncnn::Mat raw_value;
    if (extractor.extract("out0", raw_output) != 0
        || extractor.extract("out1", raw_key) != 0
        || extractor.extract("out2", raw_value) != 0) {
        return false;
    }
    output = raw_output.clone();
    present_key = raw_key.clone();
    present_value = raw_value.clone();
    return !output.empty() && !present_key.empty() && !present_value.empty();
}

bool run_single_output(
    ncnn::Net& network,
    const ncnn::Mat& input,
    ncnn::Mat& output)
{
    ncnn::Extractor extractor = network.create_extractor();
    extractor.set_light_mode(false);
    if (extractor.input("in0", input) != 0) {
        return false;
    }
    ncnn::Mat raw_output;
    if (extractor.extract("out0", raw_output) != 0) {
        return false;
    }
    output = raw_output.clone();
    return !output.empty();
}

bool run_embedding(ncnn::Net& network, int token, ncnn::Mat& output)
{
    ncnn::Mat token_input(1, static_cast<std::size_t>(4u));
    if (token_input.empty()) {
        return false;
    }
    int* token_pointer = token_input;
    token_pointer[0] = token;
    return run_single_output(network, token_input, output);
}

bool utf8_codepoint(
    const std::string& text,
    std::size_t& offset,
    std::uint32_t& codepoint)
{
    if (offset >= text.size()) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(text[offset++]);
    if (first < 0x80) {
        codepoint = first;
        return true;
    }
    int continuation_count = 0;
    if ((first & 0xE0) == 0xC0) {
        codepoint = first & 0x1F;
        continuation_count = 1;
    } else if ((first & 0xF0) == 0xE0) {
        codepoint = first & 0x0F;
        continuation_count = 2;
    } else if ((first & 0xF8) == 0xF0) {
        codepoint = first & 0x07;
        continuation_count = 3;
    } else {
        return false;
    }
    for (int index = 0; index < continuation_count; ++index) {
        if (offset >= text.size()) {
            return false;
        }
        const unsigned char next =
            static_cast<unsigned char>(text[offset++]);
        if ((next & 0xC0) != 0x80) {
            return false;
        }
        codepoint = (codepoint << 6) | (next & 0x3F);
    }
    return true;
}

bool hex_decode(const std::string& text, std::string& bytes)
{
    if (text.size() % 2 != 0) {
        return false;
    }
    bytes.clear();
    bytes.reserve(text.size() / 2);
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    for (std::size_t index = 0; index < text.size(); index += 2) {
        const int high = nibble(text[index]);
        const int low = nibble(text[index + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        bytes.push_back(static_cast<char>((high << 4) | low));
    }
    return true;
}

class ByteLevelDecoder {
public:
    bool load(const std::string& path)
    {
        std::ifstream file(path);
        std::string line;
        if (!std::getline(file, line)
            || line != "HUNYUANOCR_BYTELEVEL_VOCAB_V1") {
            return false;
        }
        if (!std::getline(file, line)) {
            return false;
        }
        const std::size_t count = static_cast<std::size_t>(std::stoul(line));
        vocabulary_.clear();
        vocabulary_.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::getline(file, line)) {
                return false;
            }
            std::string token;
            if (!hex_decode(line, token)) {
                return false;
            }
            vocabulary_.push_back(std::move(token));
        }
        build_reverse_bytes();
        return true;
    }

    bool decode(const std::vector<int>& token_ids, std::string& text) const
    {
        text.clear();
        for (int token_id : token_ids) {
            if (token_id == kEosToken) {
                continue;
            }
            if (token_id < 0
                || static_cast<std::size_t>(token_id) >= vocabulary_.size()) {
                return false;
            }
            const std::string& token = vocabulary_[token_id];
            std::size_t offset = 0;
            while (offset < token.size()) {
                std::uint32_t codepoint = 0;
                if (!utf8_codepoint(token, offset, codepoint)) {
                    return false;
                }
                const auto iterator = reverse_bytes_.find(codepoint);
                if (iterator == reverse_bytes_.end()) {
                    return false;
                }
                text.push_back(static_cast<char>(iterator->second));
            }
        }
        return true;
    }

private:
    void build_reverse_bytes()
    {
        std::vector<int> bytes;
        for (int value = 33; value <= 126; ++value) bytes.push_back(value);
        for (int value = 161; value <= 172; ++value) bytes.push_back(value);
        for (int value = 174; value <= 255; ++value) bytes.push_back(value);
        std::vector<std::uint32_t> codepoints(
            bytes.begin(), bytes.end());
        int extra = 0;
        for (int value = 0; value < 256; ++value) {
            if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
                bytes.push_back(value);
                codepoints.push_back(static_cast<std::uint32_t>(256 + extra));
                ++extra;
            }
        }
        reverse_bytes_.clear();
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            reverse_bytes_[codepoints[index]] =
                static_cast<unsigned char>(bytes[index]);
        }
    }

    std::vector<std::string> vocabulary_;
    std::unordered_map<std::uint32_t, unsigned char> reverse_bytes_;
};

bool compare_mat(
    const ncnn::Mat& actual_mat,
    const std::vector<float>& expected,
    Metrics& metrics)
{
    std::vector<float> actual;
    return unpack_mat(actual_mat, actual)
        && calculate_metrics(actual, expected, metrics);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <project_root> <packing:0|1>\n";
        return EXIT_FAILURE;
    }
    const std::string project_root = argv[1];
    const std::string packing_text = argv[2];
    if (packing_text != "0" && packing_text != "1") {
        return EXIT_FAILURE;
    }
    const bool use_packing_layout = packing_text == "1";

    std::cout << "===== HunyuanOCR ncnn prefill + generation runtime =====\n"
              << "packing layout : " << std::boolalpha
              << use_packing_layout << '\n'
              << "prefill models : 24, loaded one at a time\n"
              << "decoder models : 24, loaded after prefill\n"
              << "decode steps   : 10 maximum\n"
              << "EOS token      : " << kEosToken << "\n\n";

    ncnn::Net final_norm_network;
    ncnn::Net lm_head_network;
    ncnn::Net embedding_network;
    if (!configure_and_load(
            final_norm_network,
            project_root + "/artifacts/final_norm/final_norm.ncnn.param",
            project_root + "/artifacts/final_norm/final_norm.ncnn.bin",
            use_packing_layout)
        || !configure_and_load(
            lm_head_network,
            project_root + "/artifacts/lm_head/lm_head.ncnn.param",
            project_root + "/artifacts/lm_head/lm_head.ncnn.bin",
            use_packing_layout)
        || !configure_and_load(
            embedding_network,
            project_root + "/artifacts/text_embedding/text_embedding.ncnn.param",
            project_root + "/artifacts/text_embedding/text_embedding.ncnn.bin",
            use_packing_layout)) {
        std::cerr << "Failed to load tail or embedding network\n";
        return EXIT_FAILURE;
    }

    ByteLevelDecoder tokenizer;
    if (!tokenizer.load(
            project_root + "/artifacts/tokenizer/bytelevel_vocab.txt")) {
        std::cerr << "Failed to load exported tokenizer vocabulary\n";
        return EXIT_FAILURE;
    }

    const std::string prefill_layer0_reference =
        prefill_reference_directory(project_root, 0);
    std::vector<float> prefill_input_values;
    std::vector<float> prefill_mask_values;
    std::vector<float> prefill_rope_cos_values;
    std::vector<float> prefill_rope_sin_values;
    if (!load_exact_binary(
            prefill_layer0_reference + "/layer0_hidden_states_f32.bin",
            kPrefillHiddenCount,
            prefill_input_values)
        || !load_exact_binary(
            prefill_layer0_reference + "/layer0_attention_mask_f32.bin",
            kPrefillMaskCount,
            prefill_mask_values)
        || !load_exact_binary(
            prefill_layer0_reference
                + "/layer0_position_embeddings_0_f32.bin",
            kPrefillRopeCount,
            prefill_rope_cos_values)
        || !load_exact_binary(
            prefill_layer0_reference
                + "/layer0_position_embeddings_1_f32.bin",
            kPrefillRopeCount,
            prefill_rope_sin_values)) {
        return EXIT_FAILURE;
    }

    ncnn::Mat prefill_hidden = make_prefill_hidden(prefill_input_values);
    ncnn::Mat prefill_mask = make_prefill_mask(prefill_mask_values);
    ncnn::Mat prefill_rope_cos = make_prefill_rope(prefill_rope_cos_values);
    ncnn::Mat prefill_rope_sin = make_prefill_rope(prefill_rope_sin_values);
    std::vector<ncnn::Mat> cache_keys(kLayerCount);
    std::vector<ncnn::Mat> cache_values(kLayerCount);

    double maximum_prefill_input_error = 0.0;
    double maximum_prefill_output_error = 0.0;
    double maximum_prefill_cache_error = 0.0;

    std::cout << "===== 24-layer ncnn prefill =====\n";
    for (int layer = 0; layer < kLayerCount; ++layer) {
        const std::string prefix = "layer" + std::to_string(layer);
        const std::string reference =
            prefill_reference_directory(project_root, layer);
        std::vector<float> expected_hidden;
        std::vector<float> expected_output;
        std::vector<float> expected_key;
        std::vector<float> expected_value;
        if (!load_exact_binary(
                reference + "/" + prefix + "_hidden_states_f32.bin",
                kPrefillHiddenCount,
                expected_hidden)
            || !load_exact_binary(
                reference + "/" + prefix + "_output_f32.bin",
                kPrefillHiddenCount,
                expected_output)
            || !load_exact_binary(
                reference + "/present_key_f32.bin",
                kPrefillCacheCount,
                expected_key)
            || !load_exact_binary(
                reference + "/present_value_f32.bin",
                kPrefillCacheCount,
                expected_value)) {
            return EXIT_FAILURE;
        }

        Metrics input_metrics;
        if (!compare_mat(prefill_hidden, expected_hidden, input_metrics)
            || !prefill_hidden_passed(input_metrics)) {
            std::cerr << "Prefill layer " << layer
                      << " input parity failed: max="
                      << input_metrics.maximum_abs_error << '\n';
            return EXIT_FAILURE;
        }

        const std::string name = "decoder_layer" + std::to_string(layer)
            + "_prefill_kv";
        const std::string directory = project_root + "/artifacts/" + name;
        ncnn::Net prefill_network;
        if (!configure_and_load(
                prefill_network,
                directory + "/" + name + ".ncnn.param",
                directory + "/" + name + ".ncnn.bin",
                use_packing_layout)) {
            std::cerr << "Failed to load prefill layer " << layer << '\n';
            return EXIT_FAILURE;
        }

        ncnn::Mat next_hidden;
        ncnn::Mat present_key;
        ncnn::Mat present_value;
        if (!run_prefill_layer(
                prefill_network,
                prefill_hidden,
                prefill_mask,
                prefill_rope_cos,
                prefill_rope_sin,
                next_hidden,
                present_key,
                present_value)) {
            std::cerr << "Prefill layer " << layer << " inference failed\n";
            return EXIT_FAILURE;
        }
        prefill_network.clear();

        Metrics output_metrics;
        Metrics key_metrics;
        Metrics value_metrics;
        if (!compare_mat(next_hidden, expected_output, output_metrics)
            || !compare_mat(present_key, expected_key, key_metrics)
            || !compare_mat(present_value, expected_value, value_metrics)
            || !prefill_hidden_passed(output_metrics)
            || !prefill_cache_passed(key_metrics)
            || !prefill_cache_passed(value_metrics)) {
            std::cerr << "Prefill layer " << layer
                      << " output parity failed: hidden="
                      << output_metrics.maximum_abs_error
                      << ", key=" << key_metrics.maximum_abs_error
                      << ", value=" << value_metrics.maximum_abs_error
                      << '\n';
            return EXIT_FAILURE;
        }

        maximum_prefill_input_error = std::max(
            maximum_prefill_input_error, input_metrics.maximum_abs_error);
        maximum_prefill_output_error = std::max(
            maximum_prefill_output_error, output_metrics.maximum_abs_error);
        maximum_prefill_cache_error = std::max(
            maximum_prefill_cache_error,
            std::max(
                key_metrics.maximum_abs_error,
                value_metrics.maximum_abs_error));

        std::cout << "Prefill Layer " << std::setw(2) << layer
                  << std::scientific << std::setprecision(3)
                  << ": input_max=" << input_metrics.maximum_abs_error
                  << ", hidden_max=" << output_metrics.maximum_abs_error
                  << ", key_max=" << key_metrics.maximum_abs_error
                  << ", value_max=" << value_metrics.maximum_abs_error
                  << '\n';

        cache_keys[layer] = present_key;
        cache_values[layer] = present_value;
        prefill_hidden = next_hidden;
    }

    std::vector<float> full_prefill_hidden;
    if (!unpack_mat(prefill_hidden, full_prefill_hidden)
        || full_prefill_hidden.size() != kPrefillHiddenCount) {
        std::cerr << "Unable to unpack final prefill hidden state\n";
        return EXIT_FAILURE;
    }
    std::vector<float> last_hidden_values(
        full_prefill_hidden.end() - kHiddenSize,
        full_prefill_hidden.end());
    ncnn::Mat last_hidden = make_hidden(last_hidden_values);

    const std::string prefill_tail = prefill_tail_directory(project_root);
    std::vector<float> expected_prefill_norm_input;
    std::vector<float> expected_prefill_norm_output;
    std::vector<float> expected_prefill_logits;
    if (!load_exact_binary(
            prefill_tail + "/final_norm_input_f32.bin",
            kHiddenSize,
            expected_prefill_norm_input)
        || !load_exact_binary(
            prefill_tail + "/final_norm_output_f32.bin",
            kHiddenSize,
            expected_prefill_norm_output)
        || !load_exact_binary(
            prefill_tail + "/prefill_logits_f32.bin",
            kVocabSize,
            expected_prefill_logits)) {
        return EXIT_FAILURE;
    }

    Metrics prefill_norm_input_metrics;
    if (!compare_mat(
            last_hidden,
            expected_prefill_norm_input,
            prefill_norm_input_metrics)
        || !prefill_hidden_passed(prefill_norm_input_metrics)) {
        std::cerr << "Prefill final-norm input parity failed\n";
        return EXIT_FAILURE;
    }

    ncnn::Mat prefill_norm_output;
    ncnn::Mat prefill_logits;
    if (!run_single_output(
            final_norm_network, last_hidden, prefill_norm_output)
        || !run_single_output(
            lm_head_network, prefill_norm_output, prefill_logits)) {
        std::cerr << "Prefill tail inference failed\n";
        return EXIT_FAILURE;
    }

    Metrics prefill_norm_metrics;
    Metrics prefill_logits_metrics;
    std::vector<float> actual_prefill_logits;
    if (!compare_mat(
            prefill_norm_output,
            expected_prefill_norm_output,
            prefill_norm_metrics)
        || !unpack_mat(prefill_logits, actual_prefill_logits)
        || !calculate_metrics(
            actual_prefill_logits,
            expected_prefill_logits,
            prefill_logits_metrics)
        || !prefill_hidden_passed(prefill_norm_metrics)
        || prefill_logits_metrics.maximum_abs_error > 3.0e-3
        || prefill_logits_metrics.mean_abs_error > 5.0e-5
        || prefill_logits_metrics.cosine_similarity < 0.999999) {
        std::cerr << "Prefill tail parity failed: norm="
                  << prefill_norm_metrics.maximum_abs_error
                  << ", logits=" << prefill_logits_metrics.maximum_abs_error
                  << '\n';
        return EXIT_FAILURE;
    }

    const int prefill_token = static_cast<int>(std::distance(
        actual_prefill_logits.begin(),
        std::max_element(
            actual_prefill_logits.begin(), actual_prefill_logits.end())));
    if (prefill_token != kExpectedTokens[0]) {
        std::cerr << "Prefill token mismatch: actual=" << prefill_token
                  << ", expected=" << kExpectedTokens[0] << '\n';
        return EXIT_FAILURE;
    }

    ncnn::Mat current_hidden;
    if (!run_embedding(embedding_network, prefill_token, current_hidden)) {
        std::cerr << "Prefill token embedding failed\n";
        return EXIT_FAILURE;
    }
    std::vector<float> expected_decode_hidden;
    if (!load_exact_binary(
            reference_directory(project_root, 0, 1)
                + "/layer0_hidden_states_f32.bin",
            kHiddenSize,
            expected_decode_hidden)) {
        return EXIT_FAILURE;
    }
    Metrics prefill_embedding_metrics;
    std::vector<float> actual_decode_hidden;
    if (!unpack_mat(current_hidden, actual_decode_hidden)
        || !calculate_metrics(
            actual_decode_hidden,
            expected_decode_hidden,
            prefill_embedding_metrics)
        || actual_decode_hidden != expected_decode_hidden) {
        std::cerr << "Prefill token embedding was not byte-identical\n";
        return EXIT_FAILURE;
    }

    std::cout << "\n===== Prefill result =====\n"
              << "Prefill token        : " << prefill_token << '\n'
              << std::scientific << std::setprecision(3)
              << "Maximum input error  : " << maximum_prefill_input_error << '\n'
              << "Maximum hidden error : " << maximum_prefill_output_error << '\n'
              << "Maximum cache error  : " << maximum_prefill_cache_error << '\n'
              << "Final hidden error   : "
              << prefill_norm_input_metrics.maximum_abs_error << '\n'
              << "Final norm error     : "
              << prefill_norm_metrics.maximum_abs_error << '\n'
              << "Prefill logits error : "
              << prefill_logits_metrics.maximum_abs_error << "\n\n";

    std::vector<std::unique_ptr<ncnn::Net>> decoder_networks;
    decoder_networks.reserve(kLayerCount);
    for (int layer = 0; layer < kLayerCount; ++layer) {
        const std::string name = "decoder_layer" + std::to_string(layer)
            + "_decode_dynamic";
        const std::string directory = project_root + "/artifacts/" + name;
        auto network = std::make_unique<ncnn::Net>();
        if (!configure_and_load(
                *network,
                directory + "/" + name + ".ncnn.param",
                directory + "/" + name + ".ncnn.bin",
                use_packing_layout)) {
            std::cerr << "Failed to load dynamic decoder layer " << layer << '\n';
            return EXIT_FAILURE;
        }
        decoder_networks.push_back(std::move(network));
    }

    std::vector<int> generated_tokens = {prefill_token};
    bool reached_eos = false;

    for (int step = 1; step <= kDecodeSteps; ++step) {
        const int past_length = kPrefillLength + step - 1;
        const int present_length = kPrefillLength + step;
        const std::size_t past_count = static_cast<std::size_t>(kKvHeads)
            * past_length * kHeadDim;
        const std::size_t present_count = static_cast<std::size_t>(kKvHeads)
            * present_length * kHeadDim;

        const std::string layer0_reference =
            reference_directory(project_root, 0, step);
        std::vector<float> mask_values;
        std::vector<float> rope_cos_values;
        std::vector<float> rope_sin_values;
        if (!load_exact_binary(
                layer0_reference + "/layer0_attention_mask_f32.bin",
                present_length,
                mask_values)
            || !load_exact_binary(
                layer0_reference + "/layer0_position_embeddings_0_f32.bin",
                kRopeComponents * kHeadDim,
                rope_cos_values)
            || !load_exact_binary(
                layer0_reference + "/layer0_position_embeddings_1_f32.bin",
                kRopeComponents * kHeadDim,
                rope_sin_values)) {
            return EXIT_FAILURE;
        }
        ncnn::Mat mask = make_mask(mask_values, present_length);
        ncnn::Mat rope_cos = make_rope(rope_cos_values);
        ncnn::Mat rope_sin = make_rope(rope_sin_values);

        double maximum_input_error = 0.0;
        double maximum_output_error = 0.0;
        double maximum_cache_error = 0.0;
        double maximum_prefix_error = 0.0;

        for (int layer = 0; layer < kLayerCount; ++layer) {
            const std::string reference =
                reference_directory(project_root, layer, step);
            const std::string prefix = "layer" + std::to_string(layer);
            std::vector<float> expected_hidden;
            std::vector<float> expected_output;
            std::vector<float> expected_past_key;
            std::vector<float> expected_past_value;
            std::vector<float> expected_present_key;
            std::vector<float> expected_present_value;
            if (!load_exact_binary(
                    reference + "/" + prefix + "_hidden_states_f32.bin",
                    kHiddenSize,
                    expected_hidden)
                || !load_exact_binary(
                    reference + "/" + prefix + "_output_f32.bin",
                    kHiddenSize,
                    expected_output)
                || !load_exact_binary(
                    reference + "/past_key_f32.bin",
                    past_count,
                    expected_past_key)
                || !load_exact_binary(
                    reference + "/past_value_f32.bin",
                    past_count,
                    expected_past_value)
                || !load_exact_binary(
                    reference + "/present_key_f32.bin",
                    present_count,
                    expected_present_key)
                || !load_exact_binary(
                    reference + "/present_value_f32.bin",
                    present_count,
                    expected_present_value)) {
                return EXIT_FAILURE;
            }

            Metrics input_metrics;
            if (!compare_mat(current_hidden, expected_hidden, input_metrics)
                || !hidden_passed(input_metrics)) {
                std::cerr << "Step " << step << " layer " << layer
                          << " input parity failed\n";
                return EXIT_FAILURE;
            }
            maximum_input_error = std::max(
                maximum_input_error, input_metrics.maximum_abs_error);

            ncnn::Mat past_key = cache_keys[layer];
            ncnn::Mat past_value = cache_values[layer];
            Metrics past_key_metrics;
            Metrics past_value_metrics;
            if (!compare_mat(past_key, expected_past_key, past_key_metrics)
                || !compare_mat(
                    past_value, expected_past_value, past_value_metrics)) {
                std::cerr << "Step " << step << " layer " << layer
                          << " KV handoff comparison failed\n";
                return EXIT_FAILURE;
            }
            const bool key_handoff_passed = step == 1
                ? prefill_cache_passed(past_key_metrics)
                : cache_passed(past_key_metrics);
            const bool value_handoff_passed = step == 1
                ? prefill_cache_passed(past_value_metrics)
                : cache_passed(past_value_metrics);
            if (!key_handoff_passed || !value_handoff_passed) {
                std::cerr << "Step " << step << " layer " << layer
                          << " KV handoff parity failed: key="
                          << past_key_metrics.maximum_abs_error
                          << ", value="
                          << past_value_metrics.maximum_abs_error << '\n';
                return EXIT_FAILURE;
            }

            ncnn::Mat next_hidden;
            ncnn::Mat present_key;
            ncnn::Mat present_value;
            if (!run_decoder_layer(
                    *decoder_networks[layer],
                    current_hidden,
                    mask,
                    rope_cos,
                    rope_sin,
                    past_key,
                    past_value,
                    next_hidden,
                    present_key,
                    present_value)) {
                std::cerr << "Step " << step << " layer " << layer
                          << " inference failed\n";
                return EXIT_FAILURE;
            }

            Metrics output_metrics;
            Metrics key_metrics;
            Metrics value_metrics;
            if (!compare_mat(next_hidden, expected_output, output_metrics)
                || !compare_mat(present_key, expected_present_key, key_metrics)
                || !compare_mat(
                    present_value, expected_present_value, value_metrics)
                || !hidden_passed(output_metrics)
                || !cache_passed(key_metrics)
                || !cache_passed(value_metrics)) {
                std::cerr << "Step " << step << " layer " << layer
                          << " output parity failed: hidden="
                          << output_metrics.maximum_abs_error
                          << ", key=" << key_metrics.maximum_abs_error
                          << ", value=" << value_metrics.maximum_abs_error
                          << '\n';
                return EXIT_FAILURE;
            }
            maximum_output_error = std::max(
                maximum_output_error, output_metrics.maximum_abs_error);
            maximum_cache_error = std::max(
                maximum_cache_error,
                std::max(
                    key_metrics.maximum_abs_error,
                    value_metrics.maximum_abs_error));

            std::vector<float> actual_present_key;
            std::vector<float> actual_present_value;
            std::vector<float> actual_past_key;
            std::vector<float> actual_past_value;
            if (!unpack_mat(present_key, actual_present_key)
                || !unpack_mat(present_value, actual_present_value)
                || !unpack_mat(past_key, actual_past_key)
                || !unpack_mat(past_value, actual_past_value)) {
                return EXIT_FAILURE;
            }
            double key_prefix_error = 0.0;
            double value_prefix_error = 0.0;
            for (int head = 0; head < kKvHeads; ++head) {
                for (int position = 0; position < past_length; ++position) {
                    for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                        const std::size_t past_index =
                            (static_cast<std::size_t>(head) * past_length
                                + position) * kHeadDim + dimension;
                        const std::size_t present_index =
                            (static_cast<std::size_t>(head) * present_length
                                + position) * kHeadDim + dimension;
                        key_prefix_error = std::max(
                            key_prefix_error,
                            std::abs(
                                static_cast<double>(actual_present_key[present_index])
                                - actual_past_key[past_index]));
                        value_prefix_error = std::max(
                            value_prefix_error,
                            std::abs(
                                static_cast<double>(actual_present_value[present_index])
                                - actual_past_value[past_index]));
                    }
                }
            }
            maximum_prefix_error = std::max(
                maximum_prefix_error,
                std::max(key_prefix_error, value_prefix_error));
            if (key_prefix_error != 0.0 || value_prefix_error != 0.0) {
                std::cerr << "Step " << step << " layer " << layer
                          << " mutated the KV prefix\n";
                return EXIT_FAILURE;
            }

            cache_keys[layer] = present_key;
            cache_values[layer] = present_value;
            current_hidden = next_hidden;
        }

        const std::string tail = tail_directory(project_root, step);
        std::vector<float> expected_norm_input;
        std::vector<float> expected_norm_output;
        std::vector<float> expected_logits;
        if (!load_exact_binary(
                tail + "/final_norm_input_f32.bin",
                kHiddenSize,
                expected_norm_input)
            || !load_exact_binary(
                tail + "/final_norm_output_f32.bin",
                kHiddenSize,
                expected_norm_output)
            || !load_exact_binary(
                tail + "/decode_logits_f32.bin",
                kVocabSize,
                expected_logits)) {
            return EXIT_FAILURE;
        }
        Metrics norm_input_metrics;
        if (!compare_mat(
                current_hidden, expected_norm_input, norm_input_metrics)
            || !hidden_passed(norm_input_metrics)) {
            std::cerr << "Step " << step << " final-norm input failed\n";
            return EXIT_FAILURE;
        }

        ncnn::Mat norm_output;
        ncnn::Mat logits;
        if (!run_single_output(
                final_norm_network, current_hidden, norm_output)
            || !run_single_output(lm_head_network, norm_output, logits)) {
            std::cerr << "Step " << step << " tail inference failed\n";
            return EXIT_FAILURE;
        }
        Metrics norm_metrics;
        Metrics logits_metrics;
        std::vector<float> actual_logits;
        if (!compare_mat(norm_output, expected_norm_output, norm_metrics)
            || !unpack_mat(logits, actual_logits)
            || !calculate_metrics(
                actual_logits, expected_logits, logits_metrics)
            || norm_metrics.maximum_abs_error > 5.0e-5
            || norm_metrics.mean_abs_error > 1.0e-5
            || logits_metrics.maximum_abs_error > 3.0e-3
            || logits_metrics.mean_abs_error > 5.0e-5
            || logits_metrics.cosine_similarity < 0.999999) {
            std::cerr << "Step " << step
                      << " tail parity failed: norm_max="
                      << norm_metrics.maximum_abs_error
                      << ", norm_mean=" << norm_metrics.mean_abs_error
                      << ", logits_max="
                      << logits_metrics.maximum_abs_error
                      << ", logits_mean=" << logits_metrics.mean_abs_error
                      << ", logits_cos="
                      << logits_metrics.cosine_similarity << '\n';
            return EXIT_FAILURE;
        }
        const int actual_token = static_cast<int>(std::distance(
            actual_logits.begin(),
            std::max_element(actual_logits.begin(), actual_logits.end())));
        if (actual_token != kExpectedTokens[step]) {
            std::cerr << "Step " << step << " token mismatch: actual="
                      << actual_token << ", expected="
                      << kExpectedTokens[step] << '\n';
            return EXIT_FAILURE;
        }
        generated_tokens.push_back(actual_token);

        std::cout << "Step " << std::setw(2) << step
                  << ": input=" << kExpectedTokens[step - 1]
                  << ", output=" << actual_token
                  << ", KV=" << past_length << "->" << present_length
                  << std::scientific << std::setprecision(3)
                  << ", input_max=" << maximum_input_error
                  << ", hidden_max=" << maximum_output_error
                  << ", cache_max=" << maximum_cache_error
                  << ", logits_max=" << logits_metrics.maximum_abs_error
                  << ", prefix=" << maximum_prefix_error << '\n';

        if (actual_token == kEosToken) {
            reached_eos = true;
            break;
        }

        ncnn::Mat embedding_output;
        if (!run_embedding(embedding_network, actual_token, embedding_output)) {
            std::cerr << "Step " << step << " embedding feedback failed\n";
            return EXIT_FAILURE;
        }
        std::vector<float> expected_next_hidden;
        if (!load_exact_binary(
                reference_directory(project_root, 0, step + 1)
                    + "/layer0_hidden_states_f32.bin",
                kHiddenSize,
                expected_next_hidden)) {
            return EXIT_FAILURE;
        }
        Metrics embedding_metrics;
        std::vector<float> actual_embedding;
        if (!unpack_mat(embedding_output, actual_embedding)
            || !calculate_metrics(
                actual_embedding, expected_next_hidden, embedding_metrics)
            || embedding_metrics.maximum_abs_error != 0.0
            || actual_embedding != expected_next_hidden) {
            std::cerr << "Step " << step
                      << " embedding handoff was not byte-identical\n";
            return EXIT_FAILURE;
        }
        current_hidden = embedding_output;
    }

    if (!reached_eos
        || generated_tokens.size() != kExpectedTokens.size()
        || !std::equal(
            generated_tokens.begin(),
            generated_tokens.end(),
            kExpectedTokens.begin())) {
        std::cerr << "Generated token sequence did not reach the expected EOS\n";
        return EXIT_FAILURE;
    }

    std::string generated_text;
    if (!tokenizer.decode(generated_tokens, generated_text)) {
        std::cerr << "ByteLevel token decoding failed\n";
        return EXIT_FAILURE;
    }
    if (generated_text != kExpectedText) {
        std::cerr << "Generated text mismatch\nActual:\n"
                  << generated_text << "\nExpected:\n"
                  << kExpectedText << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "\nGenerated tokens:";
    for (int token : generated_tokens) {
        std::cout << ' ' << token;
    }
    std::cout << "\nEOS reached      : true\n"
              << "Generated text:\n"
              << generated_text << "\n\n"
              << "Full ncnn prefill + autoregressive generation passed.\n";
    return EXIT_SUCCESS;
}
