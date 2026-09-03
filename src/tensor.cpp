#include "mini_torch/tensor.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "mini_torch/tensor_iterator.h"
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

// Metadata
size_t Tensor::numel() const { return impl_->numel(); }

size_t Tensor::ndim() const { return impl_->ndim(); }

const std::vector<size_t>& Tensor::stride() const { return impl_->stride(); }

const std::vector<size_t>& Tensor::shape() const { return impl_->shape(); }

size_t Tensor::storage_offset() const { return impl_->storage_offset(); }

// Indexing
const float& Tensor::storage_at(size_t idx) const {
    if (idx >= impl_->numel()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return impl_->storage()->data_[idx];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    validate_indices(indices, shape());
    size_t index = get_storage_index(indices, stride(), storage_offset());
    return impl_->storage()->data_[index];
}

float& Tensor::storage_at(size_t idx) {
    if (idx >= numel()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return impl_->storage()->data_[idx];
}

float& Tensor::at(const std::vector<size_t>& indices) {
    validate_indices(indices, shape());
    size_t index = get_storage_index(indices, stride(), storage_offset());
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

Tensor Tensor::expand(const std::vector<size_t>& target_shape) const {
    size_t out_ndim = target_shape.size();
    if (out_ndim < ndim()) {
        throw std::invalid_argument(
            "Number of output dimension must larger than or equal to the original");
    }

    std::vector<size_t> padded_shape = left_pad_dim(shape(), out_ndim, 1);
    std::vector<size_t> padded_stride = left_pad_dim(stride(), out_ndim, 0);
    std::vector<size_t> target_stride(out_ndim);

    for (size_t i = 0; i < out_ndim; ++i) {
        if (padded_shape[i] == target_shape[i]) {
            // keep original stride
            target_stride[i] = padded_stride[i];
        } else if (padded_shape[i] == 1) {
            // broadcasted dim -> stride = 0
            target_stride[i] = 0;
        } else {
            throw std::runtime_error("Cannot broadcast original tensor to target shape");
        }
    }

    auto new_impl = std::make_shared<TensorImpl>(impl_->storage(), target_shape, target_stride,
                                                 impl_->storage_offset());

    return Tensor(new_impl);
}

// Operators
Tensor Tensor::operator+(const Tensor& tensor) const {
    std::vector<size_t> out_shape = broadcast_shape(this->shape(), tensor.shape());

    Tensor out(out_shape);

    TensorIterator iter(out, std::vector<const Tensor*>({this, &tensor}));

    while (!iter.done()) {
        std::vector<size_t> offsets = iter.current_offsets();

        out.storage_at(offsets[0]) = this->storage_at(offsets[1]) + tensor.storage_at(offsets[2]);

        iter.next();
    }

    return out;
}

Tensor Tensor::operator-(const Tensor& tensor) const {
    std::vector<size_t> out_shape = broadcast_shape(this->shape(), tensor.shape());

    Tensor out(out_shape);

    TensorIterator iter(out, std::vector<const Tensor*>({this, &tensor}));

    while (!iter.done()) {
        std::vector<size_t> offsets = iter.current_offsets();

        out.storage_at(offsets[0]) = this->storage_at(offsets[1]) - tensor.storage_at(offsets[2]);

        iter.next();
    }

    return out;
}

Tensor Tensor::operator*(const Tensor& tensor) const {
    std::vector<size_t> out_shape = broadcast_shape(this->shape(), tensor.shape());

    Tensor out(out_shape);

    TensorIterator iter(out, std::vector<const Tensor*>({this, &tensor}));

    while (!iter.done()) {
        std::vector<size_t> offsets = iter.current_offsets();

        out.storage_at(offsets[0]) = this->storage_at(offsets[1]) * tensor.storage_at(offsets[2]);

        iter.next();
    }

    return out;
}

Tensor Tensor::operator/(const Tensor& tensor) const {
    std::vector<size_t> out_shape = broadcast_shape(this->shape(), tensor.shape());

    Tensor out(out_shape);

    TensorIterator iter(out, std::vector<const Tensor*>({this, &tensor}));

    while (!iter.done()) {
        std::vector<size_t> offsets = iter.current_offsets();

        if (tensor.storage_at(offsets[2]) == 0.0f) {
            throw std::runtime_error("Cannot divide by 0");
        }

        out.storage_at(offsets[0]) = this->storage_at(offsets[1]) / tensor.storage_at(offsets[2]);

        iter.next();
    }

    return out;
}

float Tensor::sum() const {
    float sum = 0;

    IndexIterator iter(shape());

    while (!iter.done()) {
        sum += this->at(iter.index());
        iter.next();
    }

    return sum;
}

Tensor Tensor::sum(size_t dim) const {
    if (dim >= ndim()) {
        throw std::out_of_range("Dimension index out of range");
    }

    std::vector<size_t> out_shape = remove_dims(shape(), {dim});
    Tensor out(out_shape);

    IndexIterator iter(out_shape);

    while (!iter.done()) {
        std::vector<size_t> out_idx = iter.index();
        float sum = 0;

        for (size_t i = 0; i < shape()[dim]; i++) {
            std::vector<size_t> input_idx = insert_dim(out_idx, dim, i);
            sum += this->at(input_idx);
        }

        out.at(out_idx) = sum;

        iter.next();
    }

    return out;
}

float Tensor::mean() const {
    float cumsum = sum();

    float mean = cumsum / numel();

    return mean;
}

Tensor Tensor::mean(size_t dim) const {
    if (dim >= ndim()) {
        throw std::out_of_range("Dimension index out of range");
    }

    Tensor out = sum(dim) / shape()[dim];

    return out;
}

Tensor Tensor::argmax(size_t dim) const {
    if (dim >= ndim()) {
        throw std::out_of_range("Dimension index out of range");
    }

    std::vector<size_t> out_shape = remove_dims(shape(), {dim});
    Tensor out(out_shape);

    IndexIterator iter(out_shape);

    while (!iter.done()) {
        std::vector<size_t> out_idx = iter.index();
        size_t max_dim_idx = 0;
        float max_dim_value = this->at(insert_dim(out_idx, dim, 0));

        for (size_t i = 1; i < shape()[dim]; i++) {
            std::vector<size_t> input_idx = insert_dim(out_idx, dim, i);
            float curr_dim_value = this->at(input_idx);
            if (curr_dim_value > max_dim_value) {
                max_dim_idx = i;
                max_dim_value = curr_dim_value;
            };
        }

        out.at(out_idx) = max_dim_idx;

        iter.next();
    }

    return out;
}

Tensor Tensor::max(size_t dim) const {
    if (dim >= ndim()) {
        throw std::out_of_range("Dimension index out of range");
    }

    std::vector<size_t> out_shape = remove_dims(shape(), {dim});
    Tensor out(out_shape);

    IndexIterator iter(out_shape);

    while (!iter.done()) {
        std::vector<size_t> out_idx = iter.index();
        float max_dim_value = this->at(insert_dim(out_idx, dim, 0));

        for (size_t i = 1; i < shape()[dim]; i++) {
            std::vector<size_t> input_idx = insert_dim(out_idx, dim, i);
            float curr_dim_value = this->at(input_idx);
            if (curr_dim_value > max_dim_value) {
                max_dim_value = curr_dim_value;
            };
        }

        out.at(out_idx) = max_dim_value;

        iter.next();
    }

    return out;
}

Tensor Tensor::matmul(const Tensor& tensor) const {
    // Naive matmul for 2D-tensor
    if (this->ndim() != 2 || tensor.ndim() != 2) {
        throw std::invalid_argument(
            "Input tensor's shape is invalid, expect both tensors to have ndim = 2");
    }

    if (this->shape()[1] != tensor.shape()[0]) {
        throw std::invalid_argument("Input tensor's shape is invalid, expect: (M, P) and (P, N)");
    }

    std::vector<size_t> out_shape = {this->shape()[0], tensor.shape()[1]};
    Tensor out(out_shape);

    IndexIterator iter(out_shape);

    while (!iter.done()) {
        std::vector<size_t> out_idx = iter.index();
        size_t m = out_idx[0];
        size_t n = out_idx[1];
        float sum_prod = 0;

        for (size_t i = 0; i < this->shape()[1]; ++i) {
            sum_prod += this->at({m, i}) * tensor.at({i, n});
        }

        out.at(out_idx) = sum_prod;

        iter.next();
    }

    return out;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.storage_at(i) = this->storage_at(i) + scalar;
    }
    return out;
}

Tensor Tensor::operator-(float scalar) const {
    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.storage_at(i) = this->storage_at(i) - scalar;
    }
    return out;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.storage_at(i) = this->storage_at(i) * scalar;
    }
    return out;
}

Tensor Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) {
        throw std::runtime_error("Cannot divide by 0");
    }
    Tensor out(this->shape());
    for (size_t i = 0; i < this->numel(); i++) {
        out.storage_at(i) = this->storage_at(i) / scalar;
    }
    return out;
}

Tensor operator+(float scalar, const Tensor& tensor) {
    Tensor out(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); i++) {
        out.storage_at(i) = scalar + tensor.storage_at(i);
    }
    return out;
}

Tensor operator-(float scalar, const Tensor& tensor) {
    Tensor out(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); ++i) {
        out.storage_at(i) = scalar - tensor.storage_at(i);
    }
    return out;
}

Tensor operator*(float scalar, const Tensor& tensor) {
    Tensor out(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); i++) {
        out.storage_at(i) = scalar * tensor.storage_at(i);
    }
    return out;
}

Tensor operator/(float scalar, const Tensor& tensor) {
    Tensor out(tensor.shape());
    for (size_t i = 0; i < tensor.numel(); i++) {
        if (tensor.storage_at(i) == 0.0f) {
            throw std::runtime_error("Cannot divide by 0");
        }
        out.storage_at(i) = scalar / tensor.storage_at(i);
    }
    return out;
}

}  // namespace mt