#include <hunyuanocr/runtime.h>

#include "detail/model_manifest.h"
#include "detail/multimodal_input.h"
#include "detail/prompt_inputs.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <net.h>

namespace hunyuanocr {
namespace {

constexpr int kLayerCount = 24;
constexpr int kHiddenSize = 1024;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kMropeAxes = 4;
constexpr int kEosToken = 120007;

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
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
                    result[static_cast<std::size_t>(y * pack + p) * value.w
                        + static_cast<std::size_t>(x)] = source[x * pack + p];
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
                            (static_cast<std::size_t>(q * pack + p) * value.h
                                + static_cast<std::size_t>(y))
                                * value.w + static_cast<std::size_t>(x);
                        const std::size_t source_index =
                            (static_cast<std::size_t>(y) * value.w + x) * pack
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

ncnn::Mat make_hidden(std::vector<float>& values)
{
    return ncnn::Mat(kHiddenSize, 1, values.data()).clone();
}

ncnn::Mat make_prefill_hidden(std::vector<float>& values, int sequence_length)
{
    return ncnn::Mat(kHiddenSize, sequence_length, values.data()).clone();
}

ncnn::Mat make_prefill_mask(std::vector<float>& values, int sequence_length)
{
    return ncnn::Mat(
        sequence_length, sequence_length, 1, values.data()).clone();
}

ncnn::Mat make_prefill_rope(std::vector<float>& values, int sequence_length)
{
    return ncnn::Mat(
        kHeadDim, sequence_length, kMropeAxes, values.data()).clone();
}

ncnn::Mat make_decode_mask(std::vector<float>& values, int length)
{
    ncnn::Mat flat(static_cast<int>(values.size()), values.data());
    return flat.reshape(length, 1, 1).clone();
}

ncnn::Mat make_decode_rope(std::vector<float>& values)
{
    return ncnn::Mat(kHeadDim, 1, kMropeAxes, values.data()).clone();
}

bool configure_and_load(
    ncnn::Net& network,
    const std::string& model_directory,
    const std::string& name,
    const RuntimeOptions& options)
{
    network.opt.use_vulkan_compute = false;
    network.opt.use_packing_layout = options.use_packing_layout;
    network.opt.num_threads = options.num_threads;
    const std::string directory = model_directory + "/" + name;
    return network.load_param(
            (directory + "/" + name + ".ncnn.param").c_str()) == 0
        && network.load_model(
            (directory + "/" + name + ".ncnn.bin").c_str()) == 0;
}

bool run_single_output(
    ncnn::Net& network,
    const ncnn::Mat& input,
    ncnn::Mat& output)
{
    ncnn::Extractor extractor = network.create_extractor();
    extractor.set_light_mode(false);
    ncnn::Mat raw_output;
    if (extractor.input("in0", input) != 0
        || extractor.extract("out0", raw_output) != 0) {
        return false;
    }
    output = raw_output.clone();
    return !output.empty();
}

bool run_embedding(ncnn::Net& network, int token, ncnn::Mat& output)
{
    ncnn::Mat input(1, static_cast<std::size_t>(4u));
    if (input.empty()) {
        return false;
    }
    int* pointer = input;
    pointer[0] = token;
    return run_single_output(network, input, output);
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
    const int input_hidden = extractor.input("in0", hidden);
    const int input_mask = extractor.input("in1", mask);
    const int input_cos = extractor.input("in2", rope_cos);
    const int input_sin = extractor.input("in3", rope_sin);
    const int input_key = extractor.input("in4", past_key);
    const int input_value = extractor.input("in5", past_value);
    if (input_hidden != 0 || input_mask != 0 || input_cos != 0
        || input_sin != 0 || input_key != 0 || input_value != 0) {
        std::cerr << "Decoder input status: hidden=" << input_hidden
                  << ", mask=" << input_mask << ", cos=" << input_cos
                  << ", sin=" << input_sin << ", key=" << input_key
                  << ", value=" << input_value << '\n';
        return false;
    }
    ncnn::Mat raw_output;
    ncnn::Mat raw_key;
    ncnn::Mat raw_value;
    const int output_status = extractor.extract("out0", raw_output);
    const int key_status = extractor.extract("out1", raw_key);
    const int value_status = extractor.extract("out2", raw_value);
    if (output_status != 0 || key_status != 0 || value_status != 0) {
                std::cerr << "Decoder output status: hidden=" << output_status
                  << ", key=" << key_status << ", value=" << value_status
                  << " (past key dims=" << past_key.dims
                  << ", w=" << past_key.w << ", h=" << past_key.h
                  << ", c=" << past_key.c << "; hidden dims=" << hidden.dims
                  << ", w=" << hidden.w << ", h=" << hidden.h
                  << ", c=" << hidden.c << ")\n";
        return false;
    }
    output = raw_output.clone();
    present_key = raw_key.clone();
    present_value = raw_value.clone();
    return !output.empty() && !present_key.empty() && !present_value.empty();
}

bool utf8_codepoint(
    const std::string& text,
    std::size_t& offset,
    std::uint32_t& codepoint)
{
    if (offset >= text.size()) return false;
    const unsigned char first = static_cast<unsigned char>(text[offset++]);
    int continuation_count = 0;
    if ((first & 0x80) == 0) {
        codepoint = first;
    } else if ((first & 0xE0) == 0xC0) {
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
        if (offset >= text.size()) return false;
        const unsigned char next = static_cast<unsigned char>(text[offset++]);
        if ((next & 0xC0) != 0x80) return false;
        codepoint = (codepoint << 6) | (next & 0x3F);
    }
    return true;
}

bool hex_decode(const std::string& text, std::string& bytes)
{
    if (text.size() % 2 != 0) return false;
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    bytes.clear();
    bytes.reserve(text.size() / 2);
    for (std::size_t index = 0; index < text.size(); index += 2) {
        const int high = nibble(text[index]);
        const int low = nibble(text[index + 1]);
        if (high < 0 || low < 0) return false;
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
            || line != "HUNYUANOCR_BYTELEVEL_VOCAB_V1"
            || !std::getline(file, line)) {
            return false;
        }
        const std::size_t count = static_cast<std::size_t>(std::stoul(line));
        vocabulary_.clear();
        vocabulary_.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::getline(file, line)) return false;
            std::string token;
            if (!hex_decode(line, token)) return false;
            vocabulary_.push_back(std::move(token));
        }
        build_reverse_bytes();
        return true;
    }

    bool decode(const std::vector<int>& token_ids, std::string& text) const
    {
        text.clear();
        for (int token_id : token_ids) {
            if (token_id == kEosToken) continue;
            if (token_id < 0
                || static_cast<std::size_t>(token_id) >= vocabulary_.size()) {
                return false;
            }
            const std::string& token = vocabulary_[token_id];
            std::size_t offset = 0;
            while (offset < token.size()) {
                std::uint32_t codepoint = 0;
                if (!utf8_codepoint(token, offset, codepoint)) return false;
                const auto iterator = reverse_bytes_.find(codepoint);
                if (iterator == reverse_bytes_.end()) return false;
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
        std::vector<std::uint32_t> codepoints(bytes.begin(), bytes.end());
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

int argmax(const ncnn::Mat& logits, std::vector<float>& unpacked)
{
    if (!unpack_mat(logits, unpacked) || unpacked.empty()) return -1;
    return static_cast<int>(std::distance(
        unpacked.begin(),
        std::max_element(unpacked.begin(), unpacked.end())));
}

} // namespace

class Runtime::Impl {
public:
    explicit Impl(RuntimeOptions runtime_options)
        : options(std::move(runtime_options))
    {
    }

    bool load(const std::string& directory, std::string& error)
    {
        if (options.num_threads <= 0 || options.max_new_tokens <= 0) {
            error = "num_threads and max_new_tokens must be positive";
            return false;
        }
        if (!detail::verify_model_manifest(
                directory, options.manifest_verification, error)) {
            return false;
        }
        model_directory = directory;
        final_norm.clear();
        lm_head.clear();
        embedding.clear();
        decoder_networks.clear();
        if (!configure_and_load(
                final_norm, directory, "final_norm", options)) {
            error = "Unable to load final_norm model";
            return false;
        }
        if (!configure_and_load(lm_head, directory, "lm_head", options)) {
            error = "Unable to load lm_head model";
            return false;
        }
        if (!configure_and_load(
                embedding, directory, "text_embedding", options)) {
            error = "Unable to load text_embedding model";
            return false;
        }
        if (!tokenizer.load(directory + "/tokenizer/bytelevel_vocab.txt")) {
            error = "Unable to load tokenizer vocabulary";
            return false;
        }
        loaded = true;
        return true;
    }

    bool ensure_decoder_networks(std::string& error)
    {
        if (!decoder_networks.empty()) return true;
        decoder_networks.reserve(kLayerCount);
        for (int layer = 0; layer < kLayerCount; ++layer) {
            const std::string name = "decoder_layer" + std::to_string(layer)
                + "_decode_dynamic";
            auto network = std::make_unique<ncnn::Net>();
            if (!configure_and_load(
                    *network, model_directory, name, options)) {
                error = "Unable to load dynamic decoder layer "
                    + std::to_string(layer);
                decoder_networks.clear();
                return false;
            }
            decoder_networks.push_back(std::move(network));
        }
        return true;
    }

    bool recognize(
        const std::string& image_path,
        OcrResult& result,
        std::string& error)
    {
        if (!loaded) {
            error = "Runtime is not loaded";
            return false;
        }
        result = OcrResult{};
        const Clock::time_point total_start = Clock::now();
        const Clock::time_point input_start = total_start;
        MultimodalPrefillInput multimodal;
        if (!build_multimodal_prefill_input(
                model_directory,
                image_path,
                options.use_packing_layout,
                options.num_threads,
                embedding,
                multimodal)) {
            error = "Image preprocessing, vision encoding, or prompt fusion failed";
            return false;
        }
        result.original_width = multimodal.original_width;
        result.original_height = multimodal.original_height;
        result.resized_width = multimodal.resized_width;
        result.resized_height = multimodal.resized_height;
        result.image_grid_t = static_cast<int>(multimodal.image_grid_thw[0]);
        result.image_grid_h = static_cast<int>(multimodal.image_grid_thw[1]);
        result.image_grid_w = static_cast<int>(multimodal.image_grid_thw[2]);
        result.image_token_start = multimodal.image_token_start;
        result.image_token_end = multimodal.image_token_end;
        result.prefill_length = static_cast<int>(
            multimodal.prompt_inputs.input_ids.size());
        result.stats.input_seconds = seconds_between(input_start, Clock::now());

        const Clock::time_point prefill_start = Clock::now();
        std::vector<float> prefill_values = std::move(multimodal.hidden_states);
        std::vector<float> mask_values =
            std::move(multimodal.prompt_inputs.causal_mask);
        std::vector<float> cos_values =
            std::move(multimodal.prompt_inputs.rope_cos);
        std::vector<float> sin_values =
            std::move(multimodal.prompt_inputs.rope_sin);
        const int prefill_length = result.prefill_length;
        ncnn::Mat hidden = make_prefill_hidden(prefill_values, prefill_length);
        ncnn::Mat mask = make_prefill_mask(mask_values, prefill_length);
        ncnn::Mat rope_cos = make_prefill_rope(cos_values, prefill_length);
        ncnn::Mat rope_sin = make_prefill_rope(sin_values, prefill_length);
        if (hidden.empty() || mask.empty() || rope_cos.empty() || rope_sin.empty()) {
            error = "Unable to create prefill input tensors";
            return false;
        }
        std::vector<ncnn::Mat> cache_keys(kLayerCount);
        std::vector<ncnn::Mat> cache_values(kLayerCount);
        for (int layer = 0; layer < kLayerCount; ++layer) {
            const std::string name = "decoder_layer" + std::to_string(layer)
                + "_prefill_kv";
            ncnn::Net network;
            if (!configure_and_load(network, model_directory, name, options)) {
                error = "Unable to load prefill layer " + std::to_string(layer);
                return false;
            }
            ncnn::Mat next_hidden;
            if (!run_prefill_layer(
                    network,
                    hidden,
                    mask,
                    rope_cos,
                    rope_sin,
                    next_hidden,
                    cache_keys[layer],
                    cache_values[layer])) {
                error = "Prefill inference failed at layer "
                    + std::to_string(layer);
                return false;
            }
            network.clear();
            hidden = next_hidden;
        }

        std::vector<float> full_hidden;
        if (!unpack_mat(hidden, full_hidden)
            || full_hidden.size()
                != static_cast<std::size_t>(prefill_length) * kHiddenSize) {
            error = "Unable to unpack the final prefill hidden state";
            return false;
        }
        std::vector<float> last_hidden_values(
            full_hidden.end() - kHiddenSize, full_hidden.end());
        ncnn::Mat last_hidden = make_hidden(last_hidden_values);
        ncnn::Mat norm_output;
        ncnn::Mat logits;
        std::vector<float> unpacked_logits;
        if (!run_single_output(final_norm, last_hidden, norm_output)
            || !run_single_output(lm_head, norm_output, logits)) {
            error = "Prefill final norm or LM head failed";
            return false;
        }
        const int first_token = argmax(logits, unpacked_logits);
        if (first_token < 0) {
            error = "Unable to select the prefill token";
            return false;
        }
        result.token_ids.push_back(first_token);
        ncnn::Mat current_hidden;
        if (first_token != kEosToken
            && !run_embedding(embedding, first_token, current_hidden)) {
            error = "Unable to embed the prefill token";
            return false;
        }
        result.stats.prefill_seconds = seconds_between(prefill_start, Clock::now());

        const Clock::time_point decode_start = Clock::now();
        if (first_token == kEosToken) {
            result.reached_eos = true;
        } else if (!ensure_decoder_networks(error)) {
            return false;
        }
        while (!result.reached_eos
            && static_cast<int>(result.token_ids.size())
                < options.max_new_tokens) {
            const int step = static_cast<int>(result.token_ids.size());
            const int position = prefill_length + step - 1;
            const int present_length = position + 1;
            std::vector<float> decode_mask_values;
            std::vector<float> decode_cos_values;
            std::vector<float> decode_sin_values;
            if (!build_decode_position_inputs(
                    position,
                    present_length,
                    decode_mask_values,
                    decode_cos_values,
                    decode_sin_values)) {
                error = "Unable to build decode position inputs";
                return false;
            }
            ncnn::Mat decode_mask = make_decode_mask(
                decode_mask_values, present_length);
            ncnn::Mat decode_cos = make_decode_rope(decode_cos_values);
            ncnn::Mat decode_sin = make_decode_rope(decode_sin_values);
            for (int layer = 0; layer < kLayerCount; ++layer) {
                ncnn::Mat next_hidden;
                ncnn::Mat present_key;
                ncnn::Mat present_value;
                if (!run_decoder_layer(
                        *decoder_networks[layer],
                        current_hidden,
                        decode_mask,
                        decode_cos,
                        decode_sin,
                        cache_keys[layer],
                        cache_values[layer],
                        next_hidden,
                        present_key,
                        present_value)) {
                    error = "Decode inference failed for input token "
                        + std::to_string(result.token_ids.back()) + " at step "
                        + std::to_string(step) + ", layer "
                        + std::to_string(layer);
                    return false;
                }
                cache_keys[layer] = present_key;
                cache_values[layer] = present_value;
                current_hidden = next_hidden;
            }
            if (!run_single_output(final_norm, current_hidden, norm_output)
                || !run_single_output(lm_head, norm_output, logits)) {
                error = "Decode final norm or LM head failed at step "
                    + std::to_string(step);
                return false;
            }
            const int token = argmax(logits, unpacked_logits);
            if (token < 0) {
                error = "Unable to select token at decode step "
                    + std::to_string(step);
                return false;
            }
            result.token_ids.push_back(token);
            if (token == kEosToken) {
                result.reached_eos = true;
                break;
            }
            if (!run_embedding(embedding, token, current_hidden)) {
                error = "Embedding feedback failed at decode step "
                    + std::to_string(step);
                return false;
            }
        }
        result.stats.decode_seconds = seconds_between(decode_start, Clock::now());
        if (!tokenizer.decode(result.token_ids, result.text)) {
            error = "ByteLevel output decoding failed";
            return false;
        }
        result.stats.total_seconds = seconds_between(total_start, Clock::now());
        return true;
    }

    RuntimeOptions options;
    std::string model_directory;
    bool loaded = false;
    ncnn::Net final_norm;
    ncnn::Net lm_head;
    ncnn::Net embedding;
    ByteLevelDecoder tokenizer;
    std::vector<std::unique_ptr<ncnn::Net>> decoder_networks;
};

Runtime::Runtime(RuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(options)))
{
}

Runtime::~Runtime() = default;
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;

bool Runtime::load(const std::string& model_directory, std::string& error)
{
    return impl_->load(model_directory, error);
}

bool Runtime::recognize(
    const std::string& image_path,
    OcrResult& result,
    std::string& error)
{
    return impl_->recognize(image_path, result, error);
}

} // namespace hunyuanocr
