#pragma once

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

template <typename T>
void print_vector(const std::vector<T>& v, char sep = ',') {
    if (v.empty()) {
        std::cout << "Empty vector";
    } else {
        size_t last_idx = v.size() - 1;
        for (size_t i = 0; i < v.size(); i++) {
            std::cout << v[i];
            if (i != last_idx) {
                std::cout << sep << ' ';
            }
        }
    }
}

template <typename T>
bool has_duplicates(const std::vector<T>& vec) {
    std::unordered_set<T> seen;
    for (T num : vec) {
        if (!seen.insert(num).second) {
            return true;  // Insertion failed means it already exists
        }
    }
    return false;
}

inline void validate_indices(const std::vector<size_t>& indices, const std::vector<size_t>& shape) {
    size_t ndim = shape.size();
    if (indices.size() != ndim) {
        throw std::invalid_argument("Number of indices does not match number of dimension");
    }

    for (size_t dim = 0; dim < ndim; dim++) {
        if (indices[dim] >= shape[dim]) {
            throw std::out_of_range("Tensor index out of range");
        }
    }
};

inline size_t get_storage_index(const std::vector<size_t>& logical_index,
                                const std::vector<size_t>& stride, size_t storage_offset) {
    size_t storage_index = storage_offset;
    for (size_t dim = 0; dim < stride.size(); dim++) {
        storage_index += stride[dim] * logical_index[dim];
    }

    return storage_index;
};

inline std::vector<size_t> generate_dims(size_t ndim) {
    std::vector<size_t> dims(ndim);

    std::iota(dims.begin(), dims.end(), 0);

    return dims;
}

inline std::vector<size_t> find_singleton_dims(const std::vector<size_t>& shape) {
    std::vector<size_t> singleton_dims;
    size_t ndim = shape.size();
    singleton_dims.reserve(ndim);

    for (size_t dim = 0; dim < ndim; ++dim) {
        if (shape[dim] == 1) {
            singleton_dims.push_back(dim);
        }
    }

    return singleton_dims;
};

inline std::vector<size_t> remove_dims(const std::vector<size_t>& values,
                                       const std::vector<size_t>& dims) {
    std::vector<bool> mask(values.size(), false);

    for (size_t idx : dims) {
        mask[idx] = true;
    }

    std::vector<size_t> out;
    out.reserve(values.size() - dims.size());

    for (size_t i = 0; i < values.size(); i++) {
        if (!mask[i]) {
            out.push_back(values[i]);
        }
    }

    return out;
};

inline std::vector<size_t> insert_dim(const std::vector<size_t>& values, size_t dim, size_t value) {
    std::vector<size_t> out = values;
    out.insert(out.begin() + dim, value);
    return out;
};

inline std::vector<size_t> left_pad_dim(const std::vector<size_t>& values, size_t out_size,
                                        size_t value) {
    std::vector<size_t> out = values;

    while (out.size() < out_size) {
        out.insert(out.begin(), value);
    }

    return out;
};

inline std::vector<size_t> broadcast_shape(const std::vector<size_t>& shape_a,
                                           const std::vector<size_t>& shape_b) {
    size_t out_ndim = std::max(shape_a.size(), shape_b.size());
    std::vector<size_t> padded_shape_a = left_pad_dim(shape_a, out_ndim, 1);
    std::vector<size_t> padded_shape_b = left_pad_dim(shape_b, out_ndim, 1);
    std::vector<size_t> out_shape(out_ndim);

    for (size_t i = 0; i < out_ndim; ++i) {
        if ((padded_shape_a[i] == padded_shape_b[i]) || (padded_shape_a[i] == 1) ||
            (padded_shape_b[i] == 1)) {
            out_shape[i] = std::max(padded_shape_a[i], padded_shape_b[i]);
        } else {
            throw std::runtime_error("Cannot broadcast inputs");
        }
    }

    return out_shape;
};