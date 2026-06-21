#pragma once

#include <vector>

#include "tensor.h"

namespace mt {
Tensor zeros(const std::vector<size_t>& shape);
Tensor ones(const std::vector<size_t>& shape);
Tensor arange(float start, float end, float step = 1.0f);
}