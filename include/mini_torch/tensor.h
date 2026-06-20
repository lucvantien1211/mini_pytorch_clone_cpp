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