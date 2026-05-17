#include "mini_torch/tensor.h"

#include <stdexcept>

#include "mini_torch/utils.h"

namespace mt {

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<size_t>& shape) {
    this->shape_ = shape;
    size_t n_element = this->numel();

    this->data_.resize(n_element);
    for (size_t i = 0; i < n_element; i++) {
        this->data_[i] = 0;
    }
}

size_t Tensor::numel() const {
    int n_element = 1;
    for (int dim : this->shape_) {
        n_element *= dim;
    }
    return n_element;
}

const float& Tensor::at(size_t idx) const {
    if (idx > this->data_.size()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return this->data_[idx];
}

float& Tensor::at(size_t idx) {
    if (idx > this->data_.size()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return this->data_[idx];
};

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