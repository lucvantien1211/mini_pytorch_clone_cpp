#include <cmath>
#include <random>
#include <stdexcept>

#include "mini_torch/tensor.h"
#include "mini_torch/utils.h"

namespace mt {
Tensor zeros(const std::vector<size_t>& shape, bool requires_grad) {
    return Tensor(shape, requires_grad);
}

Tensor ones(const std::vector<size_t>& shape, bool requires_grad) {
    Tensor out(shape, requires_grad);
    for (size_t i = 0; i < out.numel(); ++i) {
        out.storage_at(i) = 1.0f;
    }
    return out;
}

Tensor arange(float start, float end, float step, bool requires_grad) {
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
        Tensor out({size}, requires_grad);

        out.storage_at(0) = start;
        for (size_t i = 1; i < out.numel(); ++i) {
            out.storage_at(i) = out.storage_at(i - 1) + step;
        }

        return out;
    }
}

Tensor uniform(const std::vector<size_t>& shape, float low, float high, bool requires_grad) {
    Tensor out(shape, requires_grad);

    std::uniform_real_distribution<float> dist(low, high);

    for (size_t i = 0; i < out.numel(); ++i) {
        out.storage_at(i) = dist(global_rng());
    }

    return out;
}
}  // namespace mt