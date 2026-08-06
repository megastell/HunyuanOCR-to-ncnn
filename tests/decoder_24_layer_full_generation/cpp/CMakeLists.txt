cmake_minimum_required(VERSION 3.16)

project(decoder_24_layer_full_generation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn REQUIRED)

add_executable(decoder_24_layer_full_generation main.cpp)
target_link_libraries(decoder_24_layer_full_generation PRIVATE ncnn)
target_compile_options(
    decoder_24_layer_full_generation
    PRIVATE -Wall -Wextra -Wpedantic
)
