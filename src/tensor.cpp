#include "mini_torch/tensor.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "mini_torch/utils.h"

namespace mt {
// Constructor
Tensor::Tensor() : impl_(std::make_shared<TensorImpl>()) {}

Tensor::Tensor(const std::vector<size_t>& shape) : impl_(std::make_shared<TensorImpl>(shape)) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor's shape can not be empty");
    }

    impl_->storage()->data_.resize(impl_->numel(), 0.0f);
}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

size_t Tensor::get_storage_index(const std::vector<size_t>& indices) const {
    if (indices.size() != ndim()) {
        throw std::invalid_argument("Number of indices does not match number of dimension");
    }

    size_t storage_index = 0;
    for (size_t dim = 0; dim < ndim(); dim++) {
        if (indices[dim] >= shape()[dim]) {
            throw std::out_of_range("Tensor index out of range");
        }
        storage_index += stride()[dim] * indices[dim];
    }

    return storage_index;
}

// Metadata
size_t Tensor::numel() const { return impl_->numel(); }

size_t Tensor::ndim() const { return impl_->ndim(); }

const std::vector<size_t>& Tensor::stride() const { return impl_->stride(); }

const std::vector<size_t>& Tensor::shape() const { return impl_->shape(); }

// Indexing
const float& Tensor::at(size_t idx) const {
    if (idx >= impl_->numel()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return impl_->storage()->data_[idx];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    size_t index = get_storage_index(indices);
    return impl_->storage()->data_[index];
}

float& Tensor::at(size_t idx) {
    if (idx >= numel()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return impl_->storage()->data_[idx];
}

float& Tensor::at(const std::vector<size_t>& indices) {
    size_t index = get_storage_index(indices);
    return impl_->storage()->data_[index];
}

// Print
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

// View
bool Tensor::is_contiguous() const { return impl_->is_contiguous(); }

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("Tensor is not contiguous");
    }

    if (TensorImpl::numel(new_shape) != numel()) {
        throw std::invalid_argument("Cannot reshape tensor: number of elements mismatch");
    }

    std::vector<size_t> new_stride = TensorImpl::compute_stride(new_shape);

    auto new_impl = std::make_shared<TensorImpl>(impl_->storage(), new_shape, new_stride,
                                                 impl_->storage_offset());

    return Tensor(new_impl);
}

Tensor Tensor::flatten() const { return reshape({numel()}); }

Tensor Tensor::permute(const std::vector<size_t>& dims) const {
    if (dims.size() != ndim()) {
        throw std::invalid_argument("Number of dimensions is mismatch");
    }

    if (has_duplicates(dims)) {
        throw std::invalid_argument("Duplicate dimension index");
    }

    bool all_dim_valid =
        std::all_of(dims.begin(), dims.end(), [this](size_t dim) { return dim < ndim(); });

    if (!all_dim_valid) {
        throw std::out_of_range("Dimension index out of range");
    }

    std::vector<size_t> new_shape(ndim());
    std::vector<size_t> new_stride(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
        new_shape[i] = shape()[dims[i]];
        new_stride[i] = stride()[dims[i]];
    }
    auto new_impl = std::make_shared<TensorImpl>(impl_->storage(), new_shape, new_stride,
                                                 impl_->storage_offset());

    return Tensor(new_impl);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= ndim() || dim1 >= ndim()) {
        throw std::out_of_range("Dimension index out of range");
    }
    std::vector<size_t> dims = generate_dims(ndim());
    std::swap(dims[dim0], dims[dim1]);
    return permute(dims);
}

Tensor Tensor::squeeze() const {
    std::vector<size_t> singleton_dims = find_singleton_dims(shape());
    std::vector<size_t> new_shape = remove_dims(shape(), singleton_dims);
    std::vector<size_t> new_stride = remove_dims(stride(), singleton_dims);

    auto new_impl = std::make_shared<TensorImpl>(impl_->storage(), new_shape, new_stride,
                                                 impl_->storage_offset());

    return Tensor(new_impl);
}

Tensor Tensor::squeeze(size_t dim) const {
    if (dim >= ndim()) {
        throw std::out_of_range("Dimension index out of range");
    }
    if (shape()[dim] != 1) {
        return *this;
    }

    std::vector<size_t> new_shape = remove_dims(shape(), {dim});
    std::vector<size_t> new_stride = remove_dims(stride(), {dim});
    auto new_impl = std::make_shared<TensorImpl>(impl_->storage(), new_shape, new_stride,
                                                 impl_->storage_offset());

    return Tensor(new_impl);
}

Tensor Tensor::unsqueeze(size_t dim) const {
    if (dim > ndim()) {
        throw std::out_of_range("Dimension index out of range");
    }

    std::vector<size_t> new_shape = insert_dim(shape(), dim, 1);

    size_t stride_to_add = (dim == ndim()) ? 1 : shape()[dim] * stride()[dim];

    std::vector<size_t> new_stride = insert_dim(stride(), dim, stride_to_add);

    auto new_impl = std::make_shared<TensorImpl>(impl_->storage(), new_shape, new_stride,
                                                 impl_->storage_offset());

    return Tensor(new_impl);
}

// Operators
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