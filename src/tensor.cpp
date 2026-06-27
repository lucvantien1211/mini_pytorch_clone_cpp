#include "mini_torch/tensor.h"

#include <stdexcept>

#include "mini_torch/utils.h"

namespace mt {

TensorImpl::TensorImpl() : storage_(std::make_shared<Storage>()) {}

size_t TensorImpl::numel() const { return numel(shape_); }

size_t TensorImpl::numel(const std::vector<size_t>& shape) {
    size_t n_element;
    if (shape.empty()) {
        n_element = 0;
    } else {
        n_element = 1;
        for (size_t dim : shape) {
            n_element *= dim;
        }
    }
    return n_element;
}

std::vector<size_t> TensorImpl::compute_stride(const std::vector<size_t>& shape) {
    std::vector<size_t> stride;
    if (shape.empty()) {
        stride = {};
    } else {
        size_t n_dim = shape.size();
        stride.resize(n_dim);

        stride[n_dim - 1] = 1;
        for (size_t dim = n_dim - 1; dim > 0; --dim) {
            stride[dim - 1] = shape[dim] * stride[dim];
        }
    }
    return stride;
}

Tensor::Tensor() : impl_(std::make_shared<TensorImpl>()) {
    impl_->shape_ = {};
    impl_->stride_ = TensorImpl::compute_stride(impl_->shape_);
    impl_->numel_ = impl_->numel();
}

Tensor::Tensor(const std::vector<size_t>& shape) : impl_(std::make_shared<TensorImpl>()) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor's shape can not be empty");
    }
    impl_->shape_ = shape;
    impl_->stride_ = TensorImpl::compute_stride(shape);
    impl_->numel_ = impl_->numel();

    impl_->storage_->data_.resize(impl_->numel_, 0.0f);
}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

size_t Tensor::get_linear_index(const std::vector<size_t>& indices) const {
    if (indices.size() != impl_->stride_.size()) {
        throw std::invalid_argument("Number of indices does not match number of dimension");
    }

    size_t linear_index = 0;
    for (size_t dim = 0; dim < impl_->stride_.size(); dim++) {
        if (indices[dim] >= impl_->shape_[dim]) {
            throw std::out_of_range("Tensor index out of range");
        }
        linear_index += impl_->stride_[dim] * indices[dim];
    }

    return linear_index;
}

size_t Tensor::numel() const { return impl_->numel_; }

const std::vector<size_t>& Tensor::stride() const { return impl_->stride_; }

const std::vector<size_t>& Tensor::shape() const { return impl_->shape_; }

const float& Tensor::at(size_t idx) const {
    if (idx >= impl_->numel_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return impl_->storage_->data_[idx];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    size_t index = this->get_linear_index(indices);
    return impl_->storage_->data_[index];
}

float& Tensor::at(size_t idx) {
    if (idx >= impl_->numel_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return impl_->storage_->data_[idx];
}

float& Tensor::at(const std::vector<size_t>& indices) {
    size_t index = this->get_linear_index(indices);
    return impl_->storage_->data_[index];
}

void Tensor::print() const {
    std::cout << "Tensor(\n";
    std::cout << "    shape=[";
    print_vector(impl_->shape_);
    std::cout << "],\n";
    std::cout << "    data=[";
    print_vector(impl_->storage_->data_);
    std::cout << "]\n";
    std::cout << ")";
}

Tensor Tensor::reshape(const std::vector<size_t>& shape) const {
    if (TensorImpl::numel(shape) != impl_->numel_)
        throw std::invalid_argument("Cannot reshape tensor: number of elements mismatch");

    auto new_impl = std::make_shared<TensorImpl>();

    new_impl->storage_ = impl_->storage_;

    new_impl->shape_ = shape;

    new_impl->stride_ = TensorImpl::compute_stride(shape);

    new_impl->numel_ = new_impl->numel();

    return Tensor(new_impl);
}

Tensor Tensor::flatten() const {
    Tensor flattened_tensor = this->reshape({impl_->numel_});
    return flattened_tensor;
}

Tensor Tensor::operator+(const Tensor& tensor) const {
    if (this->shape() != tensor.shape()) {
        throw std::invalid_argument("Shape mismatch");
    }

    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
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