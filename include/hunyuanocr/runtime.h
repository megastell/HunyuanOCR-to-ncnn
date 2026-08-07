#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hunyuanocr {

enum class ManifestVerification {
    none,
    size,
    sha256,
};

struct RuntimeOptions {
    bool use_packing_layout = false;
    int num_threads = 9;
    int max_new_tokens = 32;
    int max_vision_patches = 2048;
    // Retained raw decoder model bytes. Zero keeps file streaming enabled.
    int decoder_cache_budget_mib = 0;
    // Legacy mode that retains all expanded decoder networks.
    bool cache_decode_weights = false;
    ManifestVerification manifest_verification = ManifestVerification::size;
};

struct RuntimeStats {
    double input_seconds = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    double total_seconds = 0.0;
};

struct OcrResult {
    std::vector<int> token_ids;
    std::string text;
    bool reached_eos = false;
    int original_width = 0;
    int original_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int image_grid_t = 0;
    int image_grid_h = 0;
    int image_grid_w = 0;
    int image_token_start = 0;
    int image_token_end = 0;
    int prefill_length = 0;
    int resident_decoder_layers = 0;
    int memory_cached_decoder_layers = 0;
    int file_streamed_decoder_layers = 0;
    int decoder_cache_estimated_mib = 0;
    RuntimeStats stats;
};

class Runtime {
public:
    explicit Runtime(RuntimeOptions options = {});
    ~Runtime();

    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    bool load(const std::string& model_directory, std::string& error);
    bool recognize(
        const std::string& image_path,
        OcrResult& result,
        std::string& error);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hunyuanocr
