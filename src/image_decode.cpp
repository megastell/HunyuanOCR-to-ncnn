#include "detail/image_decode.h"

#include <cmath>
#include <cstddef>
#include <limits>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace hunyuanocr::detail {

bool decode_image_rgb(
    const std::string& image_path,
    std::vector<std::uint8_t>& pixels,
    int& width,
    int& height,
    std::string& error)
{
    int channels = 0;
    stbi_uc* decoded = stbi_load(
        image_path.c_str(), &width, &height, &channels, 3);
    if (decoded == nullptr) {
        error = "Unable to decode PNG/JPEG image: ";
        error += stbi_failure_reason();
        return false;
    }
    (void)channels;
    const bool valid_dimensions = width > 0 && height > 0
        && static_cast<std::size_t>(width)
            <= std::numeric_limits<std::size_t>::max()
                / static_cast<std::size_t>(height) / 3u;
    if (!valid_dimensions) {
        stbi_image_free(decoded);
        error = "Decoded image dimensions overflow the RGB buffer";
        return false;
    }
    const std::size_t value_count =
        static_cast<std::size_t>(width) * height * 3u;
    pixels.assign(decoded, decoded + value_count);
    stbi_image_free(decoded);
    return true;
}

} // namespace hunyuanocr::detail
