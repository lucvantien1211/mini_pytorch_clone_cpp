#pragma once

#include <memory>
#include <vector>

namespace mt {

// Actual data storage for tensor
class Storage {
   public:
    std::vector<float> data_;
};

// Tensor metadata
class TensorImpl {
   public:
    std::shared_ptr<Storage> storage_;
    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    size_t numel_;

    TensorImpl();

    size_t numel() const;
    static size_t numel(const std::vector<size_t>& shape);
    static std::vector<size_t> compute_stride(const std::vector<size_t>& shape);
};

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