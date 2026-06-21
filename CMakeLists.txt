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
    src/factory.cpp
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
    tests/test_tensor.cpp
)

target_link_libraries(test_tensor
    mini_torch
    GTest::GTest
    GTest::Main
)

add_test(NAME TensorTests COMMAND test_tensor)