#include <iostream>
#include <stdexcept>

#include "mini_torch/tensor.h"
#include "mini_torch/utils.h"

int main() {
    mt::Tensor t({2, 3});

    std::cout << "Initial tensor:\n";
    t.print();

    std::cout << "\n\n";

    // =====================
    // Linear indexing
    // =====================
    std::cout << "Linear indexing:\n";

    for (size_t i = 0; i < t.numel(); i++) {
        t.at(i) = static_cast<float>(i + 1);
    }

    t.print();

    std::cout << "\n\n";

    // =====================
    // Tensor metadata
    // =====================
    std::cout << "Shape: [";

    print_vector(t.shape());

    std::cout << "]\n";

    std::cout << "Numel: " << t.numel() << "\n\n";

    // =====================
    // Strides
    // =====================
    // std::cout << "Stride: [";

    // const auto& stride = t.get_stride();

    // print_vector(stride);

    // std::cout << "]\n\n";

    // =====================
    // Multi-dimensional indexing
    // =====================
    std::cout << "Tensor values using multi-dimensional indexing:\n";

    std::cout << "t({0,0}) = " << t.at({0, 0}) << '\n';
    std::cout << "t({0,1}) = " << t.at({0, 1}) << '\n';
    std::cout << "t({0,2}) = " << t.at({0, 2}) << '\n';
    std::cout << "t({1,0}) = " << t.at({1, 0}) << '\n';
    std::cout << "t({1,1}) = " << t.at({1, 1}) << '\n';
    std::cout << "t({1,2}) = " << t.at({1, 2}) << '\n';

    std::cout << "\n";

    // Modify through multidimensional indexing
    t.at({1, 2}) = 99.0f;

    std::cout << "After t.at({1,2}) = 99:\n";
    t.print();

    std::cout << "\n\n";

    // =====================
    // Linear index mapping demo
    // =====================
    // std::cout << "Linear index mapping:\n";

    // std::cout << "{0,0} -> " << t.get_linear_index({0, 0}) << '\n';

    // std::cout << "{1,2} -> " << t.get_linear_index({1, 2}) << '\n';

    // std::cout << "\n";

    // =====================
    // Exception demo
    // =====================
    try {
        std::cout << "Trying t.at({2,0})...\n";

        t.at({2, 0});
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << '\n';
    }

    return 0;
}