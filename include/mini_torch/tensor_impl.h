#pragma once

#include <memory>
#include <vector>

#include "mini_torch/autograd.h"

namespace mt {
// Actual data storage for tensor
class Storage {
   public:
    std::vector<float> data_;
};

// Tensor metadata
class TensorImpl {
   private:
    std::shared_ptr<Storage> storage_;

    std::vector<size_t> shape_;

    std::vector<size_t> stride_;

    size_t storage_offset_ = 0;

    bool requires_grad_ = false;

    bool is_leaf_ = true;

    std::shared_ptr<mt::Node> grad_fn_;

    std::shared_ptr<TensorImpl> grad_;

    void initialize_from_shape(const std::vector<size_t>& shape, bool requires_grad);

   public:
    // Constructors
    TensorImpl();

    explicit TensorImpl(const std::vector<size_t>& shape, bool requires_grad = false);

    TensorImpl(std::shared_ptr<Storage> storage, const std::vector<size_t>& shape,
               const std::vector<size_t>& stride, size_t storage_offset = 0,
               bool requires_grad = false);

    // Metadata
    size_t numel() const;
    size_t ndim() const;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& stride() const;

    size_t storage_offset() const;

    std::shared_ptr<Storage> storage() const;

    bool is_contiguous() const;

    // Gradient and Autograd
    bool requires_grad() const;

    bool is_leaf() const;

    std::shared_ptr<mt::Node> grad_fn() const;

    std::shared_ptr<TensorImpl> grad() const;

    void set_requires_grad(bool requires_grad);

    void set_grad_fn(std::shared_ptr<Node> grad_fn);

    bool has_grad() const;

    void set_grad(std::shared_ptr<TensorImpl> grad);

    // Utility
    static size_t numel(const std::vector<size_t>& shape);

    static std::vector<size_t> compute_stride(const std::vector<size_t>& shape);
};
}  // namespace mt