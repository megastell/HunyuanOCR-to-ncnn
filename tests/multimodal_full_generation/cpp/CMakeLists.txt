cmake_minimum_required(VERSION 3.16)

project(multimodal_full_generation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn REQUIRED)

add_executable(
    multimodal_full_generation
    main.cpp
    multimodal_input.cpp
)
target_include_directories(
    multimodal_full_generation
    PRIVATE ../../../third_party/stb
)
target_link_libraries(multimodal_full_generation PRIVATE ncnn)
target_compile_options(
    multimodal_full_generation
    PRIVATE -Wall -Wextra -Wpedantic
)
