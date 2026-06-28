#include "mini_torch/tensor.h"

#include <stdexcept>

#include "mini_torch/utils.h"

namespace mt {
Tensor::Tensor() : impl_(std::make_shared<TensorImpl>()) {}

Tensor::Tensor(const std::vector<size_t>& shape) : impl_(std::make_shared<TensorImpl>(shape)) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor's shape can not be empty");
    }

    impl_->storage()->data_.resize(impl_->numel(), 0.0f);
}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

size_t Tensor::get_linear_index(const std::vector<size_t>& indices) const {
    if (indices.size() != stride().size()) {
        throw std::invalid_argument("Number of indices does not match number of dimension");
    }

    size_t linear_index = 0;
    for (size_t dim = 0; dim < stride().size(); dim++) {
        if (indices[dim] >= shape()[dim]) {
            throw std::out_of_range("Tensor index out of range");
        }
        linear_index += stride()[dim] * indices[dim];
    }

    return linear_index;
}

size_t Tensor::numel() const { return impl_->numel(); }

const std::vector<size_t>& Tensor::stride() const { return impl_->stride(); }

const std::vector<size_t>& Tensor::shape() const { return impl_->shape(); }

const float& Tensor::at(size_t idx) const {
    if (idx >= impl_->numel()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return impl_->storage()->data_[idx];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    size_t index = get_linear_index(indices);
    return impl_->storage()->data_[index];
}

float& Tensor::at(size_t idx) {
    if (idx >= numel()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return impl_->storage()->data_[idx];
}

float& Tensor::at(const std::vector<size_t>& indices) {
    size_t index = get_linear_index(indices);
    return impl_->storage()->data_[index];
}

void Tensor::print() const {
    std::cout << "Tensor(\n";
    std::cout << "    shape=[";
    print_vector(shape());
    std::cout << "],\n";
    std::cout << "    data=[";
    print_vector(impl_->storage()->data_);
    std::cout << "]\n";
    std::cout << ")";
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (TensorImpl::numel(new_shape) != numel())
        throw std::invalid_argument("Cannot reshape tensor: number of elements mismatch");

    std::vector<size_t> new_stride = TensorImpl::compute_stride(new_shape);

    auto new_impl = std::make_shared<TensorImpl>(impl_->storage(), new_shape, new_stride,
                                                 impl_->storage_offset());

    return Tensor(new_impl);
}

Tensor Tensor::flatten() const {
    Tensor flattened_tensor = reshape({numel()});
    return flattened_tensor;
}

Tensor Tensor::operator+(const Tensor& tensor) const {
    if (this->shape() != tensor.shape()) {
        throw std::invalid_argument("Shape mismatch");
    }

    Tensor out(this->shape());
    for (size_t i = 0; i < numel(); i++) {
        out.at(i) = this->at(i) + tensor.at(i);
    }

    return out;
}

Tensor Tensor::operator-(const Tensor& tensor) const {
    if (this->shape() != tensor.shape()) {
        throw std::invalid_argument("Shape mismatch");
    }

    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.at(i) = this->at(i) - tensor.at(i);
    }

    return out;
}

Tensor Tensor::operator*(const Tensor& tensor) const {
    if (this->shape() != tensor.shape()) {
        throw std::invalid_argument("Shape mismatch");
    }

    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.at(i) = this->at(i) * tensor.at(i);
    }

    return out;
}

Tensor Tensor::operator/(const Tensor& tensor) const {
    if (this->shape() != tensor.shape()) {
        throw std::invalid_argument("Shape mismatch");
    }

    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        if (tensor.at(i) == 0.0f) {
            throw std::runtime_error("Cannot divide by 0");
        }
        out.at(i) = this->at(i) / tensor.at(i);
    }

    return out;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.at(i) = this->at(i) + scalar;
    }
    return out;
}

Tensor Tensor::operator-(float scalar) const {
    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.at(i) = this->at(i) - scalar;
    }
    return out;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.at(i) = this->at(i) * scalar;
    }
    return out;
}

Tensor Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) {
        throw std::runtime_error("Cannot divide by 0");
    }
    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.at(i) = this->at(i) / scalar;
    }
    return out;
}

Tensor operator+(float scalar, const Tensor& tensor) {
    Tensor out(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); i++) {
        out.at(i) = scalar + tensor.at(i);
    }
    return out;
}

Tensor operator-(float scalar, const Tensor& tensor) {
    Tensor out(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); ++i) {
        out.at(i) = scalar - tensor.at(i);
    }
    return out;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    Tensor out(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); i++) {
        out.at(i) = scalar * tensor.at(i);
    }
    return out;
}

Tensor operator/(float scalar, const Tensor& tensor) {
    Tensor out(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); i++) {
        if (tensor.at(i) == 0.0f) {
            throw std::runtime_error("Cannot divide by 0");
        }
        out.at(i) = scalar / tensor.at(i);
    }
    return out;
}

}  // namespace mt