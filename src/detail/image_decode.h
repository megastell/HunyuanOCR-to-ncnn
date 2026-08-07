#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hunyuanocr::detail {

bool decode_image_rgb(
    const std::string& image_path,
    std::vector<std::uint8_t>& pixels,
    int& width,
    int& height,
    std::string& error);

} // namespace hunyuanocr::detail
