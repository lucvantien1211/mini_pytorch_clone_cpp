#include "mini_torch/tensor.h"

#include <stdexcept>

#include "mini_torch/utils.h"

namespace mt {

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor's shape can not be empty");
    }
    this->shape_ = shape;
    this->stride_ = this->get_stride();
    size_t n_element = this->numel();

    this->data_.resize(n_element);
    for (size_t i = 0; i < n_element; i++) {
        this->data_[i] = 0;
    }
}

std::vector<size_t> Tensor::get_stride() const {
    std::vector<size_t> stride;
    size_t n_dim = this->shape_.size();
    stride.resize(n_dim);

    stride[n_dim - 1] = 1;
    for (size_t dim = n_dim - 1; dim > 0; --dim) {
        stride[dim - 1] = this->shape_[dim] * stride[dim];
    }

    return stride;
}

size_t Tensor::numel() const {
    size_t n_element = 1;
    for (size_t dim : this->shape_) {
        n_element *= dim;
    }
    return n_element;
}

size_t Tensor::get_linear_index(const std::vector<size_t>& indices) const {
    if (indices.size() != this->stride_.size()) {
        throw std::invalid_argument("Number of indices does not match number of dimension");
    }

    size_t linear_index = 0;
    for (size_t dim = 0; dim < this->stride_.size(); dim++) {
        if (indices[dim] >= this->shape_[dim]) {
            throw std::out_of_range("Tensor index out of range");
        }
        linear_index += this->stride_[dim] * indices[dim];
    }

    return linear_index;
}

const float& Tensor::at(size_t idx) const {
    if (idx >= this->data_.size()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return this->data_[idx];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    size_t index = this->get_linear_index(indices);
    return this->data_[index];
}

float& Tensor::at(size_t idx) {
    if (idx >= this->data_.size()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return this->data_[idx];
}

float& Tensor::at(const std::vector<size_t>& indices) {
    size_t index = this->get_linear_index(indices);
    return this->data_[index];
}

const std::vector<size_t>& Tensor::shape() const { return this->shape_; }

void Tensor::print() const {
    std::cout << "Tensor(\n";
    std::cout << "    shape=[";
    print_vector(this->shape_);
    std::cout << "],\n";
    std::cout << "    data=[";
    print_vector(this->data_);
    std::cout << "]\n";
    std::cout << ")";
}

}  // namespace mt