cmake_minimum_required(VERSION 3.16)

project(
    full_decode_logits_chain
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn REQUIRED)

add_executable(
    full_decode_logits_chain
    main.cpp
)

target_link_libraries(
    full_decode_logits_chain
    PRIVATE
    ncnn
)

target_compile_options(
    full_decode_logits_chain
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
