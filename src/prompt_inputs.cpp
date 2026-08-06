#include "detail/prompt_inputs.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kBosTokenId = 120000;
constexpr int kUserTokenId = 120006;
constexpr int kImageStartTokenId = 120118;
constexpr int kImageEndTokenId = 120119;
constexpr int kImageTokenId = 120120;
constexpr int kMergeSize = 2;
constexpr int kSpatialPatchSize = 1;
constexpr int kMropeAxes = 4;
constexpr int kHeadDim = 128;

const char* fixed_ocr_prompt()
{
    return u8"\u8bf7\u9010\u884c\u8bc6\u522b\u56fe\u7247\u4e2d\u7684"
        u8"\u6240\u6709\u6587\u5b57\u3002\u53ea\u8f93\u51fa\u56fe\u7247"
        u8"\u4e2d\u7684\u6587\u5b57\u672c\u8eab\uff0c\u4fdd\u7559\u6362"
        u8"\u884c\uff0c\u4e0d\u8981\u89e3\u91ca\u3002";
}

bool hex_decode(const std::string& text, std::string& bytes)
{
    if (text.size() % 2 != 0) {
        return false;
    }
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
        if (high < 0 || low < 0) {
            return false;
        }
        bytes.push_back(static_cast<char>((high << 4) | low));
    }
    return true;
}

bool next_utf8_codepoint(
    const std::string& text,
    std::size_t& offset,
    std::uint32_t& codepoint,
    std::string& encoded)
{
    if (offset >= text.size()) {
        return false;
    }
    const std::size_t start = offset;
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
        if (offset >= text.size()) {
            return false;
        }
        const unsigned char next = static_cast<unsigned char>(text[offset++]);
        if ((next & 0xC0) != 0x80) {
            return false;
        }
        codepoint = (codepoint << 6) | (next & 0x3F);
    }
    encoded.assign(text, start, offset - start);
    return true;
}

void append_utf8(std::uint32_t codepoint, std::string& text)
{
    if (codepoint <= 0x7F) {
        text.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        text.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        text.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        text.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        text.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

bool is_cjk(std::uint32_t codepoint)
{
    return (codepoint >= 0x3400 && codepoint <= 0x9FFF)
        || (codepoint >= 0x3040 && codepoint <= 0x30FF)
        || (codepoint >= 0xF900 && codepoint <= 0xFAFF);
}

class FixedOcrByteLevelBpe {
public:
    bool load(const std::string& vocabulary_path, const std::string& merges_path)
    {
        if (!load_vocabulary(vocabulary_path) || !load_merges(merges_path)) {
            return false;
        }
        build_byte_encoder();
        return true;
    }

    bool encode(const std::string& text, std::vector<std::int64_t>& ids) const
    {
        if (text != fixed_ocr_prompt()) {
            std::cerr << "The Phase 3C tokenizer only accepts the fixed OCR prompt\n";
            return false;
        }
        std::vector<std::string> pieces;
        if (!pre_tokenize(text, pieces)) {
            return false;
        }
        ids.clear();
        for (const std::string& piece : pieces) {
            std::vector<std::string> symbols;
            symbols.reserve(piece.size());
            for (unsigned char byte : piece) {
                symbols.push_back(byte_encoder_[byte]);
            }
            apply_bpe(symbols);
            for (const std::string& symbol : symbols) {
                const auto iterator = vocabulary_.find(symbol);
                if (iterator == vocabulary_.end()) {
                    std::cerr << "BPE symbol is absent from the vocabulary\n";
                    return false;
                }
                ids.push_back(iterator->second);
            }
        }
        return true;
    }

private:
    bool load_vocabulary(const std::string& path)
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
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::getline(file, line)) {
                return false;
            }
            std::string token;
            if (!hex_decode(line, token)) {
                return false;
            }
            vocabulary_[std::move(token)] = static_cast<std::int64_t>(index);
        }
        return true;
    }

    bool load_merges(const std::string& path)
    {
        std::ifstream file(path);
        std::string line;
        if (!std::getline(file, line)
            || line != "HUNYUANOCR_BYTELEVEL_BPE_MERGES_V1"
            || !std::getline(file, line)) {
            return false;
        }
        const std::size_t count = static_cast<std::size_t>(std::stoul(line));
        merge_ranks_.clear();
        for (std::size_t rank = 0; rank < count; ++rank) {
            if (!std::getline(file, line)) {
                return false;
            }
            const std::size_t separator = line.find('\t');
            if (separator == std::string::npos) {
                return false;
            }
            std::string left;
            std::string right;
            if (!hex_decode(line.substr(0, separator), left)
                || !hex_decode(line.substr(separator + 1), right)) {
                return false;
            }
            merge_ranks_[pair_key(left, right)] = rank;
        }
        return true;
    }

    void build_byte_encoder()
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
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            byte_encoder_[bytes[index]].clear();
            append_utf8(codepoints[index], byte_encoder_[bytes[index]]);
        }
    }

    static bool pre_tokenize(
        const std::string& text,
        std::vector<std::string>& pieces)
    {
        pieces.clear();
        std::string cjk_run;
        std::size_t offset = 0;
        while (offset < text.size()) {
            std::uint32_t codepoint = 0;
            std::string encoded;
            if (!next_utf8_codepoint(text, offset, codepoint, encoded)) {
                return false;
            }
            if (is_cjk(codepoint)) {
                cjk_run += encoded;
                continue;
            }
            if (!cjk_run.empty()) {
                pieces.push_back(std::move(cjk_run));
                cjk_run.clear();
            }
            pieces.push_back(std::move(encoded));
        }
        if (!cjk_run.empty()) {
            pieces.push_back(std::move(cjk_run));
        }
        return true;
    }

    static std::string pair_key(
        const std::string& left,
        const std::string& right)
    {
        std::string key = left;
        key.push_back('\0');
        key += right;
        return key;
    }

    void apply_bpe(std::vector<std::string>& symbols) const
    {
        while (symbols.size() > 1) {
            std::size_t best_rank = std::numeric_limits<std::size_t>::max();
            std::string best_left;
            std::string best_right;
            for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
                const auto iterator = merge_ranks_.find(
                    pair_key(symbols[index], symbols[index + 1]));
                if (iterator != merge_ranks_.end()
                    && iterator->second < best_rank) {
                    best_rank = iterator->second;
                    best_left = symbols[index];
                    best_right = symbols[index + 1];
                }
            }
            if (best_rank == std::numeric_limits<std::size_t>::max()) {
                break;
            }
            std::vector<std::string> merged;
            merged.reserve(symbols.size());
            for (std::size_t index = 0; index < symbols.size();) {
                if (index + 1 < symbols.size()
                    && symbols[index] == best_left
                    && symbols[index + 1] == best_right) {
                    merged.push_back(symbols[index] + symbols[index + 1]);
                    index += 2;
                } else {
                    merged.push_back(std::move(symbols[index]));
                    ++index;
                }
            }
            symbols = std::move(merged);
        }
    }

    std::unordered_map<std::string, std::int64_t> vocabulary_;
    std::unordered_map<std::string, std::size_t> merge_ranks_;
    std::string byte_encoder_[256];
};

bool validate_prompt_contract(const PromptInputs& result)
{
    static const std::vector<std::int64_t> expected_suffix = {
        120119, 1868, 4026, 537, 9977, 12858, 50812, 9738, 292,
        1537, 8287, 12858, 1843, 9738, 7812, 270, 12231, 112273,
        270, 4426, 6465, 292, 120006,
    };
    return result.input_ids.size()
            == static_cast<std::size_t>(result.image_token_end)
                + expected_suffix.size()
        && result.input_ids[0] == kBosTokenId
        && result.input_ids[1] == kImageStartTokenId
        && result.image_token_start == 2
        && result.image_token_end > result.image_token_start
        && std::all_of(
            result.input_ids.begin() + result.image_token_start,
            result.input_ids.begin() + result.image_token_end,
            [](std::int64_t token) { return token == kImageTokenId; })
        && std::equal(
            expected_suffix.begin(),
            expected_suffix.end(),
            result.input_ids.begin() + result.image_token_end);
}

void build_rope_embeddings(
    const std::vector<std::int64_t>& position_ids,
    int sequence_length,
    std::vector<float>& rope_cos,
    std::vector<float>& rope_sin)
{
    const double base = 10000.0 * std::pow(
        1000.0, static_cast<double>(kHeadDim) / (kHeadDim - 2));
    std::vector<float> inverse_frequencies(kHeadDim / 2);
    for (int index = 0; index < kHeadDim / 2; ++index) {
        const float exponent = static_cast<float>(2 * index)
            / static_cast<float>(kHeadDim);
        inverse_frequencies[index] = 1.0f / std::pow(
            static_cast<float>(base), exponent);
    }
    const std::size_t rope_count = static_cast<std::size_t>(kMropeAxes)
        * sequence_length * kHeadDim;
    rope_cos.resize(rope_count);
    rope_sin.resize(rope_count);
    for (int axis = 0; axis < kMropeAxes; ++axis) {
        for (int position = 0; position < sequence_length; ++position) {
            const float position_value = static_cast<float>(
                position_ids[
                    static_cast<std::size_t>(axis) * sequence_length
                    + position]);
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                const float frequency = position_value
                    * inverse_frequencies[dimension % (kHeadDim / 2)];
                const std::size_t index =
                    (static_cast<std::size_t>(axis) * sequence_length
                        + position)
                    * kHeadDim + dimension;
                rope_cos[index] = std::cos(frequency);
                rope_sin[index] = std::sin(frequency);
            }
        }
    }
}

} // namespace

bool build_ocr_prompt_inputs(
    const std::string& model_directory,
    const std::array<std::int64_t, 3>& image_grid_thw,
    PromptInputs& result)
{
    FixedOcrByteLevelBpe tokenizer;
    if (!tokenizer.load(
            model_directory + "/tokenizer/bytelevel_vocab.txt",
            model_directory + "/tokenizer/bytelevel_bpe_merges.txt")) {
        std::cerr << "Failed to load the C++ ByteLevel BPE assets\n";
        return false;
    }
    std::vector<std::int64_t> prompt_ids;
    if (!tokenizer.encode(fixed_ocr_prompt(), prompt_ids)) {
        return false;
    }

    const int grid_h = static_cast<int>(image_grid_thw[1]);
    const int grid_w = static_cast<int>(image_grid_thw[2]);
    if (image_grid_thw[0] != 1
        || grid_h % (kMergeSize * kSpatialPatchSize) != 0
        || grid_w % (kMergeSize * kSpatialPatchSize) != 0) {
        std::cerr << "Unsupported image grid for the OCR prompt\n";
        return false;
    }
    const int llm_h = grid_h / kMergeSize / kSpatialPatchSize;
    const int llm_w = grid_w / kMergeSize / kSpatialPatchSize;
    const int image_token_count = llm_h * (llm_w + 1) + 2;

    result = PromptInputs{};
    result.input_ids.push_back(kBosTokenId);
    result.input_ids.push_back(kImageStartTokenId);
    result.image_token_start = static_cast<int>(result.input_ids.size());
    result.input_ids.insert(
        result.input_ids.end(), image_token_count, kImageTokenId);
    result.image_token_end = static_cast<int>(result.input_ids.size());
    result.input_ids.push_back(kImageEndTokenId);
    result.input_ids.insert(
        result.input_ids.end(), prompt_ids.begin(), prompt_ids.end());
    result.input_ids.push_back(kUserTokenId);
    if (!validate_prompt_contract(result)) {
        std::cerr << "Generated prompt differs from the OCR template contract\n";
        return false;
    }

    const int sequence_length = static_cast<int>(result.input_ids.size());
    result.attention_mask.assign(sequence_length, 1);
    result.mm_token_type_ids.assign(sequence_length, 0);
    for (int position = result.image_token_start;
         position < result.image_token_end;
         ++position) {
        result.mm_token_type_ids[position] = 1;
    }

    result.position_ids.resize(
        static_cast<std::size_t>(kMropeAxes) * sequence_length);
    for (int axis = 0; axis < kMropeAxes; ++axis) {
        for (int position = 0; position < sequence_length; ++position) {
            result.position_ids[
                static_cast<std::size_t>(axis) * sequence_length + position
            ] = position;
        }
    }
    int grid_position = result.image_token_start + 1;
    for (int height = 0; height < llm_h; ++height) {
        for (int width = 0; width <= llm_w; ++width) {
            result.position_ids[sequence_length + grid_position] = width;
            result.position_ids[2 * sequence_length + grid_position] = height;
            result.position_ids[3 * sequence_length + grid_position] = 0;
            ++grid_position;
        }
    }
    if (grid_position != result.image_token_end - 1) {
        std::cerr << "Generated image mRoPE grid has the wrong length\n";
        return false;
    }

    result.causal_mask.assign(
        static_cast<std::size_t>(sequence_length) * sequence_length,
        std::numeric_limits<float>::lowest());
    for (int row = 0; row < sequence_length; ++row) {
        std::fill(
            result.causal_mask.begin()
                + static_cast<std::size_t>(row) * sequence_length,
            result.causal_mask.begin()
                + static_cast<std::size_t>(row) * sequence_length + row + 1,
            0.0f);
    }

    build_rope_embeddings(
        result.position_ids,
        sequence_length,
        result.rope_cos,
        result.rope_sin);
    return true;
}

bool build_decode_position_inputs(
    int sequence_position,
    int present_length,
    std::vector<float>& attention_mask,
    std::vector<float>& rope_cos,
    std::vector<float>& rope_sin)
{
    if (sequence_position < 0 || present_length != sequence_position + 1) {
        return false;
    }
    attention_mask.assign(present_length, 0.0f);
    std::vector<std::int64_t> position_ids(kMropeAxes, sequence_position);
    build_rope_embeddings(position_ids, 1, rope_cos, rope_sin);
    return true;
}
