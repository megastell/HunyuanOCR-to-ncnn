cmake_minimum_required(VERSION 3.16)

project(
    decoder_24_layer_two_step_chain
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn REQUIRED)

add_executable(
    decoder_24_layer_two_step_chain
    main.cpp
)

target_link_libraries(
    decoder_24_layer_two_step_chain
    PRIVATE
    ncnn
)

target_compile_options(
    decoder_24_layer_two_step_chain
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
