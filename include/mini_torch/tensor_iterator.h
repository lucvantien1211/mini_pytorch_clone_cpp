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
    IndexIterator(const std::vector<size_t>& shape);

    void next();

    bool done() const;

    const std::vector<size_t>& index() const;
};

}  // namespace mt