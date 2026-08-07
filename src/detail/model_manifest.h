#pragma once

#include <string>

#include <hunyuanocr/runtime.h>

namespace hunyuanocr::detail {

bool verify_model_manifest(
    const std::string& model_directory,
    ManifestVerification verification,
    std::string& error);

bool verify_model_compatibility(
    const std::string& model_directory,
    std::string& error);

} // namespace hunyuanocr::detail
