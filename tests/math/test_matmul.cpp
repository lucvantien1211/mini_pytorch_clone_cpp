#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

TEST(MatMulTest, MatMulProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0, 7.0).reshape({2, 3});
    mt::Tensor b = mt::arange(1.0, 7.0).reshape({3, 2});

    mt::Tensor c = a.matmul(b);

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 2}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), 22.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 28.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 49.0f);
    EXPECT_FLOAT_EQ(c.storage_at(3), 64.0f);
}

TEST(MatMulTest, MatMulProducesCorrectResultOnNonContiguousTensor) {
    mt::Tensor a = mt::arange(1.0, 7.0).reshape({2, 3});
    mt::Tensor b = mt::arange(1.0, 7.0).reshape({2, 3}).transpose(0, 1);

    mt::Tensor c = a.matmul(b);

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 2}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), 14.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 32.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 32.0f);
    EXPECT_FLOAT_EQ(c.storage_at(3), 77.0f);
}

TEST(MatMulTest, MatMulThrowsOnInvalidDimension) {
    mt::Tensor a = mt::arange(1.0, 7.0).reshape({2, 3});
    mt::Tensor b = mt::arange(1.0, 9.0).reshape({4, 2});
    mt::Tensor c = mt::ones({3, 2, 4});

    EXPECT_THROW(a.matmul(b), std::invalid_argument);
    EXPECT_THROW(a.matmul(c), std::invalid_argument);
}