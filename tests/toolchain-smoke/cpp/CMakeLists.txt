cmake_minimum_required(VERSION 3.18)

project(
    tiny_ncnn_test
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    ncnn
    CONFIG
    REQUIRED
)

add_executable(
    tiny_ncnn_test
    main.cpp
)

target_link_libraries(
    tiny_ncnn_test
    PRIVATE
    ncnn
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        tiny_ncnn_test
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()
