cmake_minimum_required(VERSION 3.18)

project(
    hunyuanocr_lm_head_parity
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn CONFIG REQUIRED)

add_executable(
    lm_head_parity
    main.cpp
)

target_link_libraries(
    lm_head_parity
    PRIVATE
    ncnn
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        lm_head_parity
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()
