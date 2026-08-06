#include <hunyuanocr/runtime.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

namespace {

void print_usage(const char* program)
{
    std::cerr
        << "Usage: " << program << " --model-dir <artifacts> --image <png>\n"
        << "       [--packing 0|1] [--threads N] [--max-new-tokens N]\n"
        << "       [--verify none|size|sha256]\n";
}

bool parse_positive_integer(const std::string& text, int& value)
{
    try {
        std::size_t consumed = 0;
        const int parsed = std::stoi(text, &consumed);
        if (consumed != text.size() || parsed <= 0) return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

long peak_resident_kib()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(
            GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
            sizeof(counters))) {
        return 0;
    }
    return static_cast<long>(counters.PeakWorkingSetSize / 1024);
#elif defined(__linux__)
    rusage usage = {};
    return getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : 0;
#else
    return 0;
#endif
}

} // namespace

int main(int argc, char** argv)
{
    std::string model_directory;
    std::string image_path;
    hunyuanocr::RuntimeOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next_value = [&]() -> const char* {
            return index + 1 < argc ? argv[++index] : nullptr;
        };
        if (argument == "--model-dir") {
            const char* value = next_value();
            if (value == nullptr) {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            model_directory = value;
        } else if (argument == "--image") {
            const char* value = next_value();
            if (value == nullptr) {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            image_path = value;
        } else if (argument == "--packing") {
            const char* value = next_value();
            if (value == nullptr
                || (std::string(value) != "0" && std::string(value) != "1")) {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            options.use_packing_layout = std::string(value) == "1";
        } else if (argument == "--threads") {
            const char* value = next_value();
            if (value == nullptr
                || !parse_positive_integer(value, options.num_threads)) {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (argument == "--max-new-tokens") {
            const char* value = next_value();
            if (value == nullptr
                || !parse_positive_integer(value, options.max_new_tokens)) {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (argument == "--verify") {
            const char* value = next_value();
            if (value == nullptr) {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
            const std::string mode = value;
            if (mode == "none") {
                options.manifest_verification =
                    hunyuanocr::ManifestVerification::none;
            } else if (mode == "size") {
                options.manifest_verification =
                    hunyuanocr::ManifestVerification::size;
            } else if (mode == "sha256") {
                options.manifest_verification =
                    hunyuanocr::ManifestVerification::sha256;
            } else {
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::cerr << "Unknown argument: " << argument << '\n';
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (model_directory.empty() || image_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const auto load_start = std::chrono::steady_clock::now();
    hunyuanocr::Runtime runtime(options);
    std::string error;
    if (!runtime.load(model_directory, error)) {
        std::cerr << "Model load failed: " << error << '\n';
        return EXIT_FAILURE;
    }
    const double load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_start).count();
    hunyuanocr::OcrResult result;
    if (!runtime.recognize(image_path, result, error)) {
        std::cerr << "OCR failed: " << error << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Model directory : " << model_directory << '\n'
              << "Input image     : " << image_path << '\n'
              << "Packing layout : " << std::boolalpha
              << options.use_packing_layout << '\n'
              << "Image size      : " << result.original_width << 'x'
              << result.original_height << " -> " << result.resized_width
              << 'x' << result.resized_height << '\n'
              << "Generated tokens:";
    for (int token : result.token_ids) std::cout << ' ' << token;
    std::cout << "\nEOS reached     : " << result.reached_eos
              << "\nGenerated text:\n" << result.text << "\n\n"
              << std::fixed << std::setprecision(3)
              << "Load seconds    : " << load_seconds << '\n'
              << "Input seconds   : " << result.stats.input_seconds << '\n'
              << "Prefill seconds : " << result.stats.prefill_seconds << '\n'
              << "Decode seconds  : " << result.stats.decode_seconds << '\n'
              << "Runtime seconds : " << result.stats.total_seconds << '\n'
              << "Peak RSS KiB    : " << peak_resident_kib() << '\n';
    return EXIT_SUCCESS;
}
