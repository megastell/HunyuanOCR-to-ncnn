cmake_minimum_required(VERSION 3.16)

project(
    decoder_layer0_two_step_kv
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn REQUIRED)

add_executable(
    decoder_layer0_two_step_kv
    main.cpp
)

target_link_libraries(
    decoder_layer0_two_step_kv
    PRIVATE
    ncnn
)

target_compile_options(
    decoder_layer0_two_step_kv
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
