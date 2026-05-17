#pragma once

#include <iostream>
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