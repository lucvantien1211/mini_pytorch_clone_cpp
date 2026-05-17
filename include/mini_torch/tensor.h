#pragma once

#include <vector>

namespace mt {
class Tensor {
   private:
    std::vector<size_t> shape_;
    std::vector<float> data_;

   public:
    // constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape);

    // methods
    size_t numel() const;
    const float& at(size_t idx) const;
    float& at(size_t idx);
    const std::vector<size_t>& shape() const;
    void print() const;
};
}  // namespace mt