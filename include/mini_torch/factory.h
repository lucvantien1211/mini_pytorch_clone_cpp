#pragma once

#include <vector>

#include "tensor.h"

namespace mt {
Tensor zeros(const std::vector<size_t>& shape, bool requires_grad = false);
Tensor ones(const std::vector<size_t>& shape, bool requires_grad = false);
Tensor arange(float start, float end, float step = 1.0f, bool requires_grad = false);
Tensor uniform(const std::vector<size_t>& shape, float low, float high, bool requires_grad = false);
}  // namespace mt