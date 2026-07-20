#pragma once

#include <iostream>
#include <numeric>
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