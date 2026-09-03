cmake_minimum_required(VERSION 3.16)

project(mini_torch LANGUAGES CXX)

# =========================================
# C++ Standard
# =========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================
# Include directories
# =========================================
include_directories(include)

# =========================================
# Library
# =========================================
add_library(mini_torch STATIC
    src/tensor.cpp
    src/tensor_impl.cpp
    src/factory.cpp
    src/tensor_iterator.cpp
)

# =========================================
# Example executable
# =========================================
add_executable(tensor_example
    examples/tensor_example.cpp
)

target_link_libraries(tensor_example
    mini_torch
)

# =========================================
# GoogleTest
# =========================================
enable_testing()

find_package(GTest REQUIRED)

add_executable(test_tensor
    tests/tensor/test_constructor.cpp
    tests/tensor/test_factory.cpp
    tests/tensor/test_indexing.cpp
    tests/tensor/test_view.cpp

    tests/math/test_binary_ops.cpp
    tests/math/test_scalar_ops.cpp
    tests/math/test_reduction.cpp
    tests/math/test_matmul.cpp
)

target_link_libraries(test_tensor
    PRIVATE
        mini_torch
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(test_tensor)