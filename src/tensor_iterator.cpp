#include "mini_torch/tensor_iterator.h"

#include <stdexcept>

#include "mini_torch/tensor.h"
#include "mini_torch/utils.h"

namespace mt {
IndexIterator::IndexIterator() {}

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

TensorIterator::TensorIterator(Tensor& output, const std::vector<const Tensor*>& inputs) {
    // Output is set to be the first operand
    operands_.push_back(make_operand(output, output.shape()));

    for (const Tensor* input : inputs) {
        operands_.push_back(make_operand(*input, output.shape()));
    }

    index_iterator_ = IndexIterator(output.shape());
}

bool TensorIterator::done() const { return index_iterator_.done(); }

void TensorIterator::next() { index_iterator_.next(); }

std::vector<size_t> TensorIterator::current_offsets() const {
    std::vector<size_t> index = index_iterator_.index();

    std::vector<size_t> offsets;

    for (Operand operand : operands_) {
        offsets.push_back(
            get_storage_index(index, operand.stride, operand.tensor->storage_offset()));
    }

    return offsets;
}

Operand make_operand(const Tensor& tensor, const std::vector<size_t>& iteration_shape) {
    return {&tensor, compute_expand_stride(tensor.shape(), tensor.stride(), iteration_shape)};
}

}  // namespace mt