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
   private:
    std::shared_ptr<Storage> storage_;

    std::vector<size_t> shape_;

    std::vector<size_t> stride_;

    size_t storage_offset_ = 0;

    void initialize_from_shape(const std::vector<size_t>& shape);

   public:
    // Constructors
    TensorImpl();

    explicit TensorImpl(const std::vector<size_t>& shape);

    TensorImpl(std::shared_ptr<Storage> storage, const std::vector<size_t>& shape,
               const std::vector<size_t>& stride, size_t storage_offset = 0);

    // Metadata
    size_t numel() const;
    size_t ndim() const;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& stride() const;

    size_t storage_offset() const;

    std::shared_ptr<Storage> storage() const;

    // Utility
    static size_t numel(const std::vector<size_t>& shape);

    static std::vector<size_t> compute_stride(const std::vector<size_t>& shape);
};
}  // namespace mt