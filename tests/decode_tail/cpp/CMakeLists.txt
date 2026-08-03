cmake_minimum_required(VERSION 3.16)

project(
    decode_tail_parity
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn REQUIRED)

add_executable(
    decode_tail_parity
    main.cpp
)

target_link_libraries(
    decode_tail_parity
    PRIVATE
    ncnn
)

target_compile_options(
    decode_tail_parity
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
