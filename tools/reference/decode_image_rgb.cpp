#include "detail/image_decode.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <png-or-jpeg> <output.ppm>\n";
        return EXIT_FAILURE;
    }
    std::vector<std::uint8_t> pixels;
    int width = 0;
    int height = 0;
    std::string error;
    if (!hunyuanocr::detail::decode_image_rgb(
            argv[1], pixels, width, height, error)) {
        std::cerr << error << '\n';
        return EXIT_FAILURE;
    }
    std::ofstream output(argv[2], std::ios::binary);
    if (!output) {
        std::cerr << "Unable to create RGB contract: " << argv[2] << '\n';
        return EXIT_FAILURE;
    }
    output << "P6\n" << width << ' ' << height << "\n255\n";
    output.write(
        reinterpret_cast<const char*>(pixels.data()),
        static_cast<std::streamsize>(pixels.size()));
    if (!output) {
        std::cerr << "Unable to write RGB contract: " << argv[2] << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "width=" << width << " height=" << height
              << " rgb_bytes=" << pixels.size() << '\n';
    return EXIT_SUCCESS;
}
