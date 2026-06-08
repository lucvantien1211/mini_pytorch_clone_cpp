#pragma once

#include <vector>

namespace mt {
class Tensor {
   private:
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    std::vector<float> data_;

    std::vector<size_t> get_stride() const;
    size_t get_linear_index(const std::vector<size_t>& indices) const;

   public:
    // constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape);

    // methods
    size_t numel() const;
    const float& at(size_t idx) const;
    const float& at(const std::vector<size_t>& indices) const;
    float& at(size_t idx);
    float& at(const std::vector<size_t>& indices);
    const std::vector<size_t>& shape() const;
    void print() const;
};
}  // namespace mt