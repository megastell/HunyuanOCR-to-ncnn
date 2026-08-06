cmake_minimum_required(VERSION 3.16)

project(hunyuanocr_runtime_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(HunyuanOCR CONFIG REQUIRED)

add_executable(runtime_api_consumer main.cpp)
target_link_libraries(runtime_api_consumer PRIVATE HunyuanOCR::runtime)
