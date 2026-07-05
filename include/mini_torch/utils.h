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