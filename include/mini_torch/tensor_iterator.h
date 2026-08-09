#pragma once

#include <vector>

#include "mini_torch/tensor.h"

namespace mt {
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
    const Tensor& out_;
    const Tensor& a_;
    const Tensor& b_;

    IndexIterator index_iterator_;

   public:
    TensorIterator(const Tensor& out, const Tensor& a, const Tensor& b);

    bool done() const;

    void next();

    size_t out_offset() const;

    size_t a_offset() const;

    size_t b_offset() const;
};

}  // namespace mt