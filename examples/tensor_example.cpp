#include <iostream>
#include <stdexcept>

#include "mini_torch/tensor.h"

int main() {
    mt::Tensor t({2, 3});

    std::cout << "Initial tensor:\n";
    t.print();

    std::cout << "\n\n";

    // modify tensor values
    t.at(0) = 1.0f;
    t.at(1) = 2.0f;
    t.at(2) = 3.0f;

    std::cout << "After modification:\n";
    t.print();

    std::cout << "\n\n";

    // tensor metadata
    std::cout << "numel: " << t.numel() << '\n';

    std::cout << "shape: [";
    for (size_t i = 0; i < t.shape().size(); i++) {
        std::cout << t.shape()[i];

        if (i + 1 != t.shape().size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n\n";

    // exception demo
    try {
        std::cout << "Trying out-of-range access...\n";

        t.at(100);

    } catch (const std::out_of_range& e) {
        std::cout << "Caught exception: " << e.what() << '\n';
    }

    return 0;
}