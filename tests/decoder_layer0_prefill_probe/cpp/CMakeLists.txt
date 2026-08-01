cmake_minimum_required(VERSION 3.18)

project(
    hunyuanocr_decoder_layer0_prefill_probe
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn CONFIG REQUIRED)

add_executable(
    decoder_layer0_prefill_probe
    main.cpp
)

target_link_libraries(
    decoder_layer0_prefill_probe
    PRIVATE
    ncnn
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        decoder_layer0_prefill_probe
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()
