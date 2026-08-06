cmake_minimum_required(VERSION 3.16)

project(vision_tower_full LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn REQUIRED)

add_executable(vision_tower_full main.cpp)
target_link_libraries(vision_tower_full PRIVATE ncnn)
target_compile_options(
    vision_tower_full
    PRIVATE -Wall -Wextra -Wpedantic
)
