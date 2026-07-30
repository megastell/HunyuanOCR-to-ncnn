#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <net.h>

namespace {

bool load_expected(
    const std::string& file_path,
    std::vector<float>& expected_values)
{
    std::ifstream input_file(file_path);

    if (!input_file.is_open()) {
        std::cerr
            << "错误：无法打开 PyTorch 参考结果文件："
            << file_path
            << '\n';

        return false;
    }

    float value = 0.0f;

    while (input_file >> value) {
        expected_values.push_back(value);
    }

    if (expected_values.empty()) {
        std::cerr
            << "错误：参考结果文件为空："
            << file_path
            << '\n';

        return false;
    }

    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr
            << "用法：\n"
            << argv[0]
            << " <tiny.ncnn.param>"
            << " <tiny.ncnn.bin>"
            << " <expected.txt>\n";

        return EXIT_FAILURE;
    }

    const std::string param_path = argv[1];
    const std::string model_path = argv[2];
    const std::string expected_path = argv[3];

    ncnn::Net network;

    // 当前测试只使用 CPU。
    network.opt.use_vulkan_compute = false;

    // 关闭张量打包，方便直接读取简单测试模型的输出。
    network.opt.use_packing_layout = false;

    // 先采用较保守的 9 个线程。
    network.opt.num_threads = 9;

    const int load_param_result =
        network.load_param(param_path.c_str());

    if (load_param_result != 0) {
        std::cerr
            << "错误：加载 ncnn param 文件失败，返回值："
            << load_param_result
            << "\n文件："
            << param_path
            << '\n';

        return EXIT_FAILURE;
    }

    const int load_model_result =
        network.load_model(model_path.c_str());

    if (load_model_result != 0) {
        std::cerr
            << "错误：加载 ncnn bin 文件失败，返回值："
            << load_model_result
            << "\n文件："
            << model_path
            << '\n';

        return EXIT_FAILURE;
    }

    // 与 Python 中完全相同的输入：
    // [[1.0, -2.0, 0.5, 3.0]]
    ncnn::Mat input_tensor(4);

    input_tensor[0] = 1.0f;
    input_tensor[1] = -2.0f;
    input_tensor[2] = 0.5f;
    input_tensor[3] = 3.0f;

    ncnn::Extractor extractor = network.create_extractor();


    const int input_result =
        extractor.input("in0", input_tensor);

    if (input_result != 0) {
        std::cerr
            << "错误：写入输入节点 in0 失败，返回值："
            << input_result
            << '\n';

        return EXIT_FAILURE;
    }

    ncnn::Mat output_tensor;

    const int extract_result =
        extractor.extract("out0", output_tensor);

    if (extract_result != 0) {
        std::cerr
            << "错误：提取输出节点 out0 失败，返回值："
            << extract_result
            << '\n';

        return EXIT_FAILURE;
    }

    if (output_tensor.empty()) {
        std::cerr << "错误：ncnn 输出张量为空。\n";
        return EXIT_FAILURE;
    }

    std::vector<float> expected_values;

    if (!load_expected(expected_path, expected_values)) {
        return EXIT_FAILURE;
    }

    // 本测试模型的输出应为一维向量。
    // total() 可能包含为 SIMD 对齐而分配的尾部存储空间，
    // 不能用它表示逻辑元素数量。
    if (output_tensor.dims != 1) {
        std::cerr
            << "错误：预期一维输出，但实际 dims="
            << output_tensor.dims
            << '\n';

        return EXIT_FAILURE;
    }

    const std::size_t actual_count =
        static_cast<std::size_t>(output_tensor.w) *
        static_cast<std::size_t>(output_tensor.elempack);

    std::cout
        << "ncnn 输出形状："
        << "dims=" << output_tensor.dims
        << ", w=" << output_tensor.w
        << ", h=" << output_tensor.h
        << ", d=" << output_tensor.d
        << ", c=" << output_tensor.c
        << ", elempack=" << output_tensor.elempack
        << ", cstep=" << output_tensor.cstep
        << ", total(storage)=" << output_tensor.total()
        << '\n';

    if (actual_count != expected_values.size()) {
        std::cerr
            << "错误：输出元素数量不一致。\n"
            << "ncnn 元素数："
            << actual_count
            << '\n'
            << "PyTorch 元素数："
            << expected_values.size()
            << '\n';

        return EXIT_FAILURE;
    }

    const float* actual_values = output_tensor;

    float max_absolute_error = 0.0f;

    std::cout << std::fixed << std::setprecision(9);

    std::cout << "PyTorch expected: [";

    for (std::size_t index = 0;
         index < expected_values.size();
         ++index) {
        if (index != 0) {
            std::cout << ", ";
        }

        std::cout << expected_values[index];
    }

    std::cout << "]\n";

    std::cout << "ncnn output     : [";

    for (std::size_t index = 0;
         index < actual_count;
         ++index) {
        if (index != 0) {
            std::cout << ", ";
        }

        const float actual_value = actual_values[index];

        std::cout << actual_value;

        const float absolute_error =
            std::fabs(
                actual_value -
                expected_values[index]);

        max_absolute_error =
            std::max(
                max_absolute_error,
                absolute_error);
    }

    std::cout << "]\n";

    std::cout
        << "最大绝对误差："
        << max_absolute_error
        << '\n';

    constexpr float tolerance = 1e-5f;

    if (max_absolute_error > tolerance) {
        std::cerr
            << "❌ 失败：PyTorch 与 ncnn 的误差超过 "
            << tolerance
            << '\n';

        return EXIT_FAILURE;
    }

    std::cout
        << "✅ 成功："
        << "PyTorch → pnnx → ncnn → C++ CPU "
        << "完整链路已经打通。\n";

    return EXIT_SUCCESS;
}
