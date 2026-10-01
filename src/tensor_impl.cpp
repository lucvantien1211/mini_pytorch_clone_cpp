#include "mini_torch/tensor_impl.h"

#include <stdexcept>

namespace mt {
// Constructor
TensorImpl::TensorImpl() { initialize_from_shape({}, false); }

TensorImpl::TensorImpl(const std::vector<size_t>& shape, bool requires_grad) {
    initialize_from_shape(shape, requires_grad);
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage, const std::vector<size_t>& shape,
                       const std::vector<size_t>& stride, size_t storage_offset, bool requires_grad)
    : storage_(storage),
      shape_(shape),
      stride_(stride),
      storage_offset_(storage_offset),
      requires_grad_(requires_grad) {
    is_leaf_ = true;
    grad_fn_ = nullptr;
}

void TensorImpl::initialize_from_shape(const std::vector<size_t>& shape, bool requires_grad) {
    storage_ = std::make_shared<Storage>();

    shape_ = shape;
    stride_ = compute_stride(shape);

    requires_grad_ = requires_grad;

    is_leaf_ = true;

    grad_fn_ = nullptr;
}

void TensorImpl::set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

void TensorImpl::set_grad_fn(std::shared_ptr<Node> grad_fn) {
    grad_fn_ = std::move(grad_fn);
    is_leaf_ = (grad_fn_ == nullptr);
}

// Metadata
size_t TensorImpl::numel() const { return numel(shape_); }

size_t TensorImpl::ndim() const { return shape_.size(); }

const std::vector<size_t>& TensorImpl::shape() const { return shape_; }

const std::vector<size_t>& TensorImpl::stride() const { return stride_; }

size_t TensorImpl::storage_offset() const { return storage_offset_; }

std::shared_ptr<Storage> TensorImpl::storage() const { return storage_; }

bool TensorImpl::is_contiguous() const { return TensorImpl::compute_stride(shape_) == stride_; }

bool TensorImpl::requires_grad() const { return requires_grad_; }

bool TensorImpl::is_leaf() const { return is_leaf_; }

std::shared_ptr<mt::Node> TensorImpl::grad_fn() const { return grad_fn_; }

// Utility
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

}  // namespace mt