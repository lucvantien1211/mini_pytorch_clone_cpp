#pragma once

#include <vector>

#include "mini_torch/tensor.h"

namespace mt {
struct Operand {
    const Tensor* tensor;
    std::vector<size_t> stride;
};

class IndexIterator {
   private:
    std::vector<size_t> iteration_shape_;
    std::vector<size_t> current_index_;

    bool finished_;

    void increment_index();

   public:
    IndexIterator();
    IndexIterator(const std::vector<size_t>& shape);

    void next();

    bool done() const;

    const std::vector<size_t>& index() const;
};

class TensorIterator {
   private:
    std::vector<Operand> operands_;

    IndexIterator index_iterator_;

   public:
    TensorIterator(Tensor& output, const std::vector<const Tensor*>& inputs);

    bool done() const;

    void next();

    std::vector<size_t> current_offsets() const;
};

Operand make_operand(const Tensor& tensor, const std::vector<size_t>& iteration_shape);

}  // namespace mt