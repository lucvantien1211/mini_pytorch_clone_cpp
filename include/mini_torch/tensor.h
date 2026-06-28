#pragma once

#include <memory>
#include <vector>

#include "mini_torch/tensor_impl.h"

namespace mt {
class Tensor {
   private:
    std::shared_ptr<TensorImpl> impl_;

    size_t get_linear_index(const std::vector<size_t>& indices) const;

   public:
    // constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    // methods
    size_t numel() const;
    const std::vector<size_t>& stride() const;
    const std::vector<size_t>& shape() const;
    const float& at(size_t idx) const;
    const float& at(const std::vector<size_t>& indices) const;
    float& at(size_t idx);
    float& at(const std::vector<size_t>& indices);
    void print() const;

    Tensor reshape(const std::vector<size_t>& shape) const;
    Tensor flatten() const;

    // operators
    // element-wise tensor operators
    Tensor operator+(const Tensor& tensor) const;
    Tensor operator-(const Tensor& tensor) const;
    Tensor operator*(const Tensor& tensor) const;
    Tensor operator/(const Tensor& tensor) const;

    // tensor - scalar operators for: tensor + - * / scalar
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
};

// tensor - scalar operators for: scalar + - * / tensor
Tensor operator+(float scalar, const Tensor& tensor);
Tensor operator-(float scalar, const Tensor& tensor);
Tensor operator*(float scalar, const Tensor& tensor);
Tensor operator/(float scalar, const Tensor& tensor);

}  // namespace mt