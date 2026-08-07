cmake_minimum_required(VERSION 3.16)

project(HunyuanOCR_ncnn VERSION 0.1.0 LANGUAGES CXX)

option(HUNYUANOCR_BUILD_CLI "Build the HunyuanOCR command-line application" ON)
option(HUNYUANOCR_BUILD_PARITY_TESTS "Build reference-backed parity tests" OFF)
option(HUNYUANOCR_BUILD_BENCHMARKS "Build runtime benchmark utilities" OFF)
option(HUNYUANOCR_ENABLE_RELEASE_TESTS "Register full OCR release tests with CTest" OFF)
option(
    HUNYUANOCR_ENABLE_REPRODUCTION_TESTS
    "Register slow reproducible artifact generation acceptance with CTest"
    OFF
)
set(HUNYUANOCR_RELEASE_MODEL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/artifacts"
    CACHE PATH "Model directory used by HunyuanOCR release CTest entries"
)
set(HUNYUANOCR_RELEASE_LOG_DIR
    "$ENV{HOME}/hunyuanocr-recovery/phase4f/ctest"
    CACHE PATH "Persistent log directory used by HunyuanOCR release CTest entries"
)
set(HUNYUANOCR_REPRO_HF_MODEL_DIR
    "$ENV{HOME}/work/hunyuanocr/models/HunyuanOCR-1.5"
    CACHE PATH "Local HuggingFace model directory used by reproducible artifact CTest"
)
set(HUNYUANOCR_REPRO_WORK_DIR
    "$ENV{HOME}/hunyuanocr-recovery/phase4k"
    CACHE PATH "Persistent staging directory used by reproducible artifact CTest"
)
set(HUNYUANOCR_REPRO_REFERENCE_PYTHON
    "$ENV{HOME}/work/hunyuanocr/.venv-reference/bin/python"
    CACHE FILEPATH "Reference Python used by reproducible artifact CTest"
)
set(HUNYUANOCR_REPRO_PNNX
    "$ENV{HOME}/work/hunyuanocr/.venv-pnnx/bin/pnnx"
    CACHE FILEPATH "pnnx executable used by reproducible artifact CTest"
)
set(HUNYUANOCR_REPRO_REPORT
    "${CMAKE_CURRENT_SOURCE_DIR}/docs/phase4k_reproducible_release_acceptance.json"
    CACHE FILEPATH "Phase 4K reproducible release acceptance report"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ncnn CONFIG REQUIRED)

add_library(
    hunyuanocr_runtime
    src/image_decode.cpp
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
target_compile_definitions(
    hunyuanocr_runtime
    PRIVATE HUNYUANOCR_RUNTIME_VERSION="${PROJECT_VERSION}"
)

if(MSVC)
    target_compile_options(hunyuanocr_runtime PRIVATE /W4 /permissive-)
else()
    target_compile_options(
        hunyuanocr_runtime PRIVATE -Wall -Wextra -Wpedantic
    )
endif()

if(HUNYUANOCR_BUILD_BENCHMARKS)
    add_executable(
        hunyuanocr_repeat_benchmark
        tools/benchmark/runtime_repeat_benchmark.cpp
    )
    target_link_libraries(
        hunyuanocr_repeat_benchmark PRIVATE HunyuanOCR::runtime
    )
    add_executable(
        hunyuanocr_decode_image_rgb
        tools/reference/decode_image_rgb.cpp
    )
    target_include_directories(
        hunyuanocr_decode_image_rgb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(
        hunyuanocr_decode_image_rgb PRIVATE HunyuanOCR::runtime
    )
    if(MSVC)
        target_link_libraries(hunyuanocr_repeat_benchmark PRIVATE Psapi)
        target_compile_options(
            hunyuanocr_repeat_benchmark PRIVATE /W4 /permissive-
        )
        target_compile_options(
            hunyuanocr_decode_image_rgb PRIVATE /W4 /permissive-
        )
    else()
        target_compile_options(
            hunyuanocr_repeat_benchmark PRIVATE -Wall -Wextra -Wpedantic
        )
        target_compile_options(
            hunyuanocr_decode_image_rgb PRIVATE -Wall -Wextra -Wpedantic
        )
    endif()
endif()

if(HUNYUANOCR_BUILD_CLI)
    add_executable(hunyuanocr_cli app/hunyuanocr_cli.cpp)
    target_link_libraries(hunyuanocr_cli PRIVATE HunyuanOCR::runtime)
    if(MSVC)
        target_link_libraries(hunyuanocr_cli PRIVATE Psapi)
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

include(CTest)
if(HUNYUANOCR_ENABLE_RELEASE_TESTS AND HUNYUANOCR_BUILD_CLI)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    file(MAKE_DIRECTORY "${HUNYUANOCR_RELEASE_LOG_DIR}")
    set(HUNYUANOCR_CTEST_COMMON
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/validate/validate_runtime_suite.py"
        --cli "$<TARGET_FILE:hunyuanocr_cli>"
        --model-dir "${HUNYUANOCR_RELEASE_MODEL_DIR}"
        --log-dir "${HUNYUANOCR_RELEASE_LOG_DIR}"
        --verify size
    )
    function(hunyuanocr_add_release_test test_name suite_name)
        add_test(
            NAME "${test_name}"
            COMMAND ${HUNYUANOCR_CTEST_COMMON} --suite "${suite_name}"
        )
        set_tests_properties(
            "${test_name}"
            PROPERTIES
                LABELS "hunyuanocr;release;${suite_name}"
                TIMEOUT 3600
        )
    endfunction()
    hunyuanocr_add_release_test(hunyuanocr_smoke_exact smoke)
    hunyuanocr_add_release_test(hunyuanocr_dynamic_sizes_exact dynamic)
    hunyuanocr_add_release_test(hunyuanocr_real_png_exact real-png)
    hunyuanocr_add_release_test(hunyuanocr_real_jpeg_exact real-jpeg)
    hunyuanocr_add_release_test(hunyuanocr_decoder_cache_budgets cache-budgets)
    hunyuanocr_add_release_test(hunyuanocr_error_paths error-paths)
endif()

if(HUNYUANOCR_ENABLE_REPRODUCTION_TESTS)
    if(NOT HUNYUANOCR_BUILD_CLI)
        message(FATAL_ERROR "HUNYUANOCR_ENABLE_REPRODUCTION_TESTS requires HUNYUANOCR_BUILD_CLI")
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME hunyuanocr_reproduce_runtime_artifacts_acceptance
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/release/reproduce_runtime_artifacts_acceptance.py"
            --clean-staging
            --hf-model-dir "${HUNYUANOCR_REPRO_HF_MODEL_DIR}"
            --reference-python "${HUNYUANOCR_REPRO_REFERENCE_PYTHON}"
            --pnnx "${HUNYUANOCR_REPRO_PNNX}"
            --cli "$<TARGET_FILE:hunyuanocr_cli>"
            --work-dir "${HUNYUANOCR_REPRO_WORK_DIR}"
            --report "${HUNYUANOCR_REPRO_REPORT}"
    )
    set_tests_properties(
        hunyuanocr_reproduce_runtime_artifacts_acceptance
        PROPERTIES
            LABELS "hunyuanocr;release;reproducible-artifacts;slow"
            TIMEOUT 14400
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
    FILES
        LICENSE
        NOTICE
        README.md
        THIRD_PARTY_NOTICES.md
        docs/install_linux.md
        docs/install_windows.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)
install(
    DIRECTORY third_party/licenses/
    DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party/licenses
)
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

set(CPACK_PACKAGE_NAME "HunyuanOCR-ncnn")
set(CPACK_PACKAGE_VENDOR "HunyuanOCR-ncnn contributors")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "https://github.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "HunyuanOCR ncnn runtime library and OCR CLI"
)
set(CPACK_PACKAGE_FILE_NAME
    "HunyuanOCR-ncnn-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
)
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)
