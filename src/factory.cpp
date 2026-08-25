#include <cmath>
#include <stdexcept>

#include "mini_torch/tensor.h"

namespace mt {
Tensor zeros(const std::vector<size_t>& shape) { return Tensor(shape); }

Tensor ones(const std::vector<size_t>& shape) {
    Tensor out(shape);
    for (size_t i = 0; i < out.numel(); ++i) {
        out.storage_at(i) = 1.0f;
    }
    return out;
}

Tensor arange(float start, float end, float step = 1.0f) {
    if (step == 0.0f) {
        throw std::invalid_argument("Step argument must not equal 0.0");
    }

    double span = static_cast<double>(end) - start;
    double n = span / step;

    constexpr double eps = 1e-12;
    size_t size = static_cast<size_t>(std::max(0.0, std::ceil(n - eps)));

    if (size == 0) {
        return Tensor();
    } else {
        Tensor out({size});

        out.storage_at(0) = start;
        for (size_t i = 1; i < out.numel(); ++i) {
            out.storage_at(i) = out.storage_at(i - 1) + step;
        }

        return out;
    }
}
}  // namespace mt