#include <hunyuanocr/runtime.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace {

long current_resident_kib()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    return GetProcessMemoryInfo(
               GetCurrentProcess(),
               reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
               sizeof(counters))
        ? static_cast<long>(counters.WorkingSetSize / 1024)
        : 0;
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    long total_pages = 0;
    long resident_pages = 0;
    if (!(statm >> total_pages >> resident_pages)) return 0;
    (void)total_pages;
    return resident_pages * sysconf(_SC_PAGESIZE) / 1024;
#else
    return 0;
#endif
}

long peak_resident_kib()
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    counters.cb = sizeof(counters);
    return GetProcessMemoryInfo(
               GetCurrentProcess(),
               reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
               sizeof(counters))
        ? static_cast<long>(counters.PeakWorkingSetSize / 1024)
        : 0;
#elif defined(__linux__)
    rusage usage = {};
    return getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : 0;
#else
    return 0;
#endif
}

bool parse_positive(const char* text, int& value)
{
    try {
        std::size_t consumed = 0;
        const int parsed = std::stoi(text, &consumed);
        if (text[consumed] != '\0' || parsed <= 0) return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_non_negative(const char* text, int& value)
{
    if (std::string(text) == "0") {
        value = 0;
        return true;
    }
    return parse_positive(text, value);
}

std::string token_csv(const std::vector<int>& tokens)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) output << ',';
        output << tokens[index];
    }
    return output.str();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model-dir> <packing:0|1> <threads> <iterations>"
                  << " [--decoder-cache-mib N] <image> [image...]\n";
        return EXIT_FAILURE;
    }
    const std::string model_directory = argv[1];
    const std::string packing = argv[2];
    int threads = 0;
    int iterations = 0;
    if ((packing != "0" && packing != "1")
        || !parse_positive(argv[3], threads)
        || !parse_positive(argv[4], iterations)) {
        return EXIT_FAILURE;
    }
    hunyuanocr::RuntimeOptions options;
    options.use_packing_layout = packing == "1";
    options.num_threads = threads;
    options.manifest_verification = hunyuanocr::ManifestVerification::size;
    std::vector<std::string> images;
    for (int index = 5; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--decoder-cache-mib") {
            if (index + 1 >= argc
                || !parse_non_negative(
                    argv[++index], options.decoder_cache_budget_mib)) {
                return EXIT_FAILURE;
            }
        } else {
            images.push_back(argument);
        }
    }
    if (images.empty()) return EXIT_FAILURE;
    hunyuanocr::Runtime runtime(options);
    std::string error;
    if (!runtime.load(model_directory, error)) {
        std::cerr << "load_error=" << error << '\n';
        return EXIT_FAILURE;
    }
    const long initial_rss = current_resident_kib();
    std::cout << "initial_rss_kib=" << initial_rss << '\n';
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const std::string& image = images[
            static_cast<std::size_t>(iteration) % images.size()];
        const auto start = std::chrono::steady_clock::now();
        hunyuanocr::OcrResult result;
        if (!runtime.recognize(image, result, error)) {
            std::cerr << "iteration=" << iteration + 1
                      << " error=" << error << '\n';
            return EXIT_FAILURE;
        }
        const double wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "iteration=" << iteration + 1
                  << " image=" << image
                  << " grid=1," << result.image_grid_h << ','
                  << result.image_grid_w
                  << " resident_layers=" << result.resident_decoder_layers
                  << " memory_layers="
                  << result.memory_cached_decoder_layers
                  << " file_layers=" << result.file_streamed_decoder_layers
                  << " cache_estimated_mib="
                  << result.decoder_cache_estimated_mib
                  << " tokens=" << token_csv(result.token_ids)
                  << " runtime_seconds=" << result.stats.total_seconds
                  << " wall_seconds=" << wall_seconds
                  << " current_rss_kib=" << current_resident_kib()
                  << " peak_rss_kib=" << peak_resident_kib() << '\n';
    }
    std::cout << "final_rss_kib=" << current_resident_kib() << '\n'
              << "peak_rss_kib=" << peak_resident_kib() << '\n';
    return EXIT_SUCCESS;
}
