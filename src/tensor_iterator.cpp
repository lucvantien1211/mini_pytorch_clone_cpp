#include "mini_torch/tensor_iterator.h"

#include <stdexcept>

#include "mini_torch/tensor.h"

namespace mt {
IndexIterator::IndexIterator(const std::vector<size_t>& shape) {
    iteration_shape_ = shape;

    current_index_ = std::vector<size_t>(iteration_shape_.size(), 0);

    finished_ = shape.empty() ? true : false;
}

void IndexIterator::next() { increment_index(); }

bool IndexIterator::done() const { return finished_; }

const std::vector<size_t>& IndexIterator::index() const { return current_index_; }

void IndexIterator::increment_index() {
    if (finished_) return;

    // Iterate from last dimension
    for (int i = static_cast<int>(iteration_shape_.size()) - 1; i >= 0; --i) {
        ++current_index_[i];

        if (current_index_[i] < iteration_shape_[i]) {
            // Valid index
            return;
        }

        // Overflow -> reset and carry on to next dimension
        current_index_[i] = 0;
    }

    // Finish iterating
    finished_ = true;
}
}  // namespace mt