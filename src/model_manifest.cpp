#include "detail/model_manifest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef HUNYUANOCR_RUNTIME_VERSION
#define HUNYUANOCR_RUNTIME_VERSION "0.0.0"
#endif

namespace hunyuanocr::detail {
namespace {

constexpr const char* kManifestHeader =
    "HUNYUANOCR_NCNN_RUNTIME_MANIFEST_V1";
constexpr const char* kCompatibilityHeader =
    "HUNYUANOCR_NCNN_RUNTIME_COMPATIBILITY_V1";

constexpr std::array<std::uint32_t, 64> kSha256Constants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

std::uint32_t rotate_right(std::uint32_t value, int amount)
{
    return (value >> amount) | (value << (32 - amount));
}

void transform_sha256(
    const unsigned char* block,
    std::array<std::uint32_t, 8>& state)
{
    std::uint32_t words[64] = {};
    for (int index = 0; index < 16; ++index) {
        words[index] = (static_cast<std::uint32_t>(block[index * 4]) << 24)
            | (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16)
            | (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8)
            | static_cast<std::uint32_t>(block[index * 4 + 3]);
    }
    for (int index = 16; index < 64; ++index) {
        const std::uint32_t s0 = rotate_right(words[index - 15], 7)
            ^ rotate_right(words[index - 15], 18)
            ^ (words[index - 15] >> 3);
        const std::uint32_t s1 = rotate_right(words[index - 2], 17)
            ^ rotate_right(words[index - 2], 19)
            ^ (words[index - 2] >> 10);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (int index = 0; index < 64; ++index) {
        const std::uint32_t s1 = rotate_right(e, 6)
            ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t choice = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + s1 + choice
            + kSha256Constants[index] + words[index];
        const std::uint32_t s0 = rotate_right(a, 2)
            ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

bool sha256_file(const std::filesystem::path& path, std::string& digest)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    std::array<std::uint32_t, 8> state = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<unsigned char, 64> block = {};
    std::uint64_t total_bytes = 0;
    while (file) {
        file.read(
            reinterpret_cast<char*>(block.data()),
            static_cast<std::streamsize>(block.size()));
        const std::streamsize count = file.gcount();
        if (count == static_cast<std::streamsize>(block.size())) {
            transform_sha256(block.data(), state);
            total_bytes += block.size();
            continue;
        }
        total_bytes += static_cast<std::uint64_t>(count);
        std::size_t offset = static_cast<std::size_t>(count);
        block[offset++] = 0x80;
        if (offset > 56) {
            std::fill(block.begin() + static_cast<std::ptrdiff_t>(offset), block.end(), 0);
            transform_sha256(block.data(), state);
            block.fill(0);
            offset = 0;
        }
        std::fill(
            block.begin() + static_cast<std::ptrdiff_t>(offset),
            block.begin() + 56,
            0);
        const std::uint64_t total_bits = total_bytes * 8;
        for (int index = 0; index < 8; ++index) {
            block[63 - index] = static_cast<unsigned char>(
                total_bits >> (index * 8));
        }
        transform_sha256(block.data(), state);
        break;
    }
    if (file.bad()) {
        return false;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::uint32_t value : state) {
        output << std::setw(8) << value;
    }
    digest = output.str();
    return true;
}

struct ManifestEntry {
    std::string relative_path;
    std::uintmax_t bytes = 0;
    std::string sha256;
};

bool parse_manifest(
    const std::filesystem::path& path,
    std::vector<ManifestEntry>& entries,
    std::string& error)
{
    std::ifstream file(path);
    std::string line;
    if (!std::getline(file, line) || line != kManifestHeader) {
        error = "Invalid runtime manifest header: " + path.string();
        return false;
    }
    if (!std::getline(file, line)) {
        error = "Runtime manifest is missing file_count";
        return false;
    }
    const std::size_t separator = line.find('\t');
    if (separator == std::string::npos
        || line.substr(0, separator) != "file_count") {
        error = "Invalid runtime manifest file_count";
        return false;
    }
    std::size_t expected_count = 0;
    try {
        expected_count = static_cast<std::size_t>(
            std::stoull(line.substr(separator + 1)));
    } catch (...) {
        error = "Invalid runtime manifest file count value";
        return false;
    }
    entries.clear();
    while (std::getline(file, line)) {
        const std::size_t first = line.find('\t');
        const std::size_t second = first == std::string::npos
            ? std::string::npos : line.find('\t', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            error = "Invalid runtime manifest entry";
            return false;
        }
        ManifestEntry entry;
        entry.relative_path = line.substr(0, first);
        entry.sha256 = line.substr(second + 1);
        try {
            entry.bytes = static_cast<std::uintmax_t>(
                std::stoull(line.substr(first + 1, second - first - 1)));
        } catch (...) {
            error = "Invalid size for manifest entry: " + entry.relative_path;
            return false;
        }
        const std::filesystem::path relative(entry.relative_path);
        if (relative.empty() || relative.is_absolute()
            || entry.relative_path.find("..") != std::string::npos
            || entry.sha256.size() != 64) {
            error = "Unsafe or invalid manifest entry: " + entry.relative_path;
            return false;
        }
        entries.push_back(std::move(entry));
    }
    if (entries.size() != expected_count) {
        error = "Runtime manifest file count mismatch";
        return false;
    }
    return true;
}

std::vector<int> parse_version(const std::string& text)
{
    std::vector<int> values;
    std::size_t offset = 0;
    while (offset < text.size()) {
        const std::size_t dot = text.find('.', offset);
        const std::string part = text.substr(
            offset,
            dot == std::string::npos ? std::string::npos : dot - offset);
        if (part.empty()
            || part.find_first_not_of("0123456789") != std::string::npos) {
            return {};
        }
        values.push_back(std::stoi(part));
        if (dot == std::string::npos) break;
        offset = dot + 1;
    }
    while (values.size() < 3) values.push_back(0);
    return values;
}

int compare_versions(const std::string& left, const std::string& right)
{
    const std::vector<int> lhs = parse_version(left);
    const std::vector<int> rhs = parse_version(right);
    if (lhs.empty() || rhs.empty()) return 0;
    const std::size_t count = std::max(lhs.size(), rhs.size());
    for (std::size_t index = 0; index < count; ++index) {
        const int a = index < lhs.size() ? lhs[index] : 0;
        const int b = index < rhs.size() ? rhs[index] : 0;
        if (a < b) return -1;
        if (a > b) return 1;
    }
    return 0;
}

bool parse_compatibility(
    const std::filesystem::path& path,
    std::map<std::string, std::string>& values,
    std::string& error)
{
    std::ifstream file(path);
    std::string line;
    if (!std::getline(file, line) || line != kCompatibilityHeader) {
        error = "Invalid runtime compatibility header: " + path.string();
        return false;
    }
    values.clear();
    while (std::getline(file, line)) {
        const std::size_t separator = line.find('\t');
        if (separator == std::string::npos || separator == 0
            || separator + 1 == line.size()) {
            error = "Invalid runtime compatibility entry";
            return false;
        }
        values[line.substr(0, separator)] = line.substr(separator + 1);
    }
    return true;
}

bool require_value(
    const std::map<std::string, std::string>& values,
    const std::string& key,
    std::string& value,
    std::string& error)
{
    const auto iterator = values.find(key);
    if (iterator == values.end()) {
        error = "Runtime compatibility is missing key: " + key;
        return false;
    }
    value = iterator->second;
    return true;
}

} // namespace

bool verify_model_manifest(
    const std::string& model_directory,
    ManifestVerification verification,
    std::string& error)
{
    if (verification == ManifestVerification::none) {
        return true;
    }
    const std::filesystem::path root =
        std::filesystem::weakly_canonical(model_directory);
    std::vector<ManifestEntry> entries;
    if (!parse_manifest(root / "runtime_manifest.tsv", entries, error)) {
        return false;
    }
    for (const ManifestEntry& entry : entries) {
        const std::filesystem::path path = root / entry.relative_path;
        std::error_code status_error;
        if (!std::filesystem::is_regular_file(path, status_error)) {
            error = "Missing model file: " + path.string();
            return false;
        }
        if (std::filesystem::file_size(path, status_error) != entry.bytes
            || status_error) {
            error = "Model file size mismatch: " + path.string();
            return false;
        }
        if (verification == ManifestVerification::sha256) {
            std::string actual;
            if (!sha256_file(path, actual)) {
                error = "Unable to hash model file: " + path.string();
                return false;
            }
            if (actual != entry.sha256) {
                error = "Model SHA-256 mismatch: " + path.string();
                return false;
            }
        }
    }
    return true;
}

bool verify_model_compatibility(
    const std::string& model_directory,
    std::string& error)
{
    const std::filesystem::path root =
        std::filesystem::weakly_canonical(model_directory);
    const std::filesystem::path path = root / "runtime_compatibility.tsv";
    std::error_code status;
    if (!std::filesystem::is_regular_file(path, status)) {
        return true;
    }
    std::map<std::string, std::string> values;
    if (!parse_compatibility(path, values, error)) {
        return false;
    }
    std::string value;
    if (!require_value(values, "model_id", value, error)) return false;
    if (value != "tencent/HunyuanOCR") {
        error = "Unsupported model_id in runtime compatibility: " + value;
        return false;
    }
    if (!require_value(values, "manifest_format", value, error)) return false;
    if (value != kManifestHeader) {
        error = "Unsupported manifest format in runtime compatibility: "
            + value;
        return false;
    }
    if (!require_value(values, "runtime_min_version", value, error)) {
        return false;
    }
    const std::string runtime_version = HUNYUANOCR_RUNTIME_VERSION;
    if (compare_versions(runtime_version, value) < 0) {
        error = "Runtime " + runtime_version
            + " is older than model minimum runtime " + value;
        return false;
    }
    if (!require_value(values, "runtime_max_exclusive_version", value, error)) {
        return false;
    }
    if (compare_versions(runtime_version, value) >= 0) {
        error = "Runtime " + runtime_version
            + " is newer than model maximum exclusive runtime " + value;
        return false;
    }
    if (!require_value(values, "runtime_abi_major", value, error)) {
        return false;
    }
    const std::vector<int> version = parse_version(runtime_version);
    if (version.empty() || std::to_string(version[0]) != value) {
        error = "Runtime ABI major mismatch: runtime " + runtime_version
            + ", model ABI " + value;
        return false;
    }
    if (!require_value(values, "file_count", value, error)) return false;
    std::vector<ManifestEntry> entries;
    if (!parse_manifest(root / "runtime_manifest.tsv", entries, error)) {
        return false;
    }
    if (value != std::to_string(entries.size())) {
        error = "Runtime compatibility file_count does not match manifest";
        return false;
    }
    if (!require_value(values, "jpeg_pixel_contract", value, error)) {
        return false;
    }
    if (value != "stb_rgb_v1") {
        error = "Unsupported JPEG pixel contract: " + value;
        return false;
    }
    return true;
}

} // namespace hunyuanocr::detail
