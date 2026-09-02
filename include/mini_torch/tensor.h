#pragma once

#include <memory>
#include <vector>

#include "mini_torch/tensor_impl.h"

namespace mt {
class Tensor {
   private:
    std::shared_ptr<TensorImpl> impl_;

   public:
    // Constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    explicit Tensor(std::shared_ptr<TensorImpl> impl);

    // Methods
    // Metadata
    size_t numel() const;
    size_t ndim() const;
    const std::vector<size_t>& stride() const;
    const std::vector<size_t>& shape() const;
    size_t storage_offset() const;

    // Indexing
    const float& storage_at(size_t idx) const;
    const float& at(const std::vector<size_t>& indices) const;
    float& storage_at(size_t idx);
    float& at(const std::vector<size_t>& indices);

    // Printing
    void print() const;

    // View
    bool is_contiguous() const;
    Tensor contiguous() const;
    Tensor reshape(const std::vector<size_t>& shape) const;
    Tensor flatten() const;
    Tensor permute(const std::vector<size_t>& dims) const;
    Tensor transpose(size_t dim0, size_t dim1) const;
    Tensor squeeze() const;
    Tensor squeeze(size_t dim) const;
    Tensor unsqueeze(size_t dim) const;
    Tensor expand(const std::vector<size_t>& target_shape) const;

    // Operators
    // Element-wise tensor operators
    Tensor operator+(const Tensor& tensor) const;
    Tensor operator-(const Tensor& tensor) const;
    Tensor operator*(const Tensor& tensor) const;
    Tensor operator/(const Tensor& tensor) const;

    // Tensor - scalar operators for: tensor + - * / scalar
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // Reduction
    float sum() const;
    Tensor sum(size_t dim) const;
    float mean() const;
    Tensor mean(size_t dim) const;
    Tensor argmax(size_t dim) const;
    Tensor max(size_t dim) const;
};

// Tensor - scalar operators for: scalar + - * / tensor
Tensor operator+(float scalar, const Tensor& tensor);
Tensor operator-(float scalar, const Tensor& tensor);
Tensor operator*(float scalar, const Tensor& tensor);
Tensor operator/(float scalar, const Tensor& tensor);

}  // namespace mt