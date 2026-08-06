cmake_minimum_required(VERSION 3.16)

project(HunyuanOCR_ncnn VERSION 0.1.0 LANGUAGES CXX)

option(HUNYUANOCR_BUILD_CLI "Build the HunyuanOCR command-line application" ON)
option(HUNYUANOCR_BUILD_PARITY_TESTS "Build reference-backed parity tests" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn CONFIG REQUIRED)

add_library(
    hunyuanocr_runtime
    src/model_manifest.cpp
    src/multimodal_input.cpp
    src/prompt_inputs.cpp
    src/runtime.cpp
)
add_library(HunyuanOCR::runtime ALIAS hunyuanocr_runtime)
set_target_properties(hunyuanocr_runtime PROPERTIES EXPORT_NAME runtime)
target_include_directories(
    hunyuanocr_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(hunyuanocr_runtime PUBLIC ncnn)

if(MSVC)
    target_compile_options(hunyuanocr_runtime PRIVATE /W4 /permissive-)
else()
    target_compile_options(
        hunyuanocr_runtime PRIVATE -Wall -Wextra -Wpedantic
    )
endif()

if(HUNYUANOCR_BUILD_CLI)
    add_executable(hunyuanocr_cli app/hunyuanocr_cli.cpp)
    target_link_libraries(hunyuanocr_cli PRIVATE HunyuanOCR::runtime)
    if(MSVC)
        target_compile_options(hunyuanocr_cli PRIVATE /W4 /permissive-)
    else()
        target_compile_options(
            hunyuanocr_cli PRIVATE -Wall -Wextra -Wpedantic
        )
    endif()
endif()

if(HUNYUANOCR_BUILD_PARITY_TESTS)
    add_subdirectory(
        tests/multimodal_full_generation/cpp
        tests/multimodal_full_generation
    )
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/HunyuanOCRConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HunyuanOCRConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HunyuanOCR
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/HunyuanOCRConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    TARGETS hunyuanocr_runtime
    EXPORT HunyuanOCRTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(HUNYUANOCR_BUILD_CLI)
    install(TARGETS hunyuanocr_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT HunyuanOCRTargets
    NAMESPACE HunyuanOCR::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HunyuanOCR
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/HunyuanOCRConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/HunyuanOCRConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HunyuanOCR
)
