cmake_minimum_required(VERSION 3.16)

project(
    decoder_two_layer_decode_parity
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn REQUIRED)

add_executable(
    decoder_two_layer_decode_parity
    main.cpp
)

target_link_libraries(
    decoder_two_layer_decode_parity
    PRIVATE
    ncnn
)

target_compile_options(
    decoder_two_layer_decode_parity
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
