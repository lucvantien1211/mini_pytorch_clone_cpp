#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

TEST(UnaryOpsTest, ReluProducesCorrectResult) {
    // Input tensor a: [[1 0 3], [4 5 -6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    a.storage_at(1) = 0;
    a.storage_at(5) = -6;
    mt::Tensor b = a.relu();

    // shape
    EXPECT_EQ(b.shape(), std::vector<size_t>({2, 3}));

    // value
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(b.storage_at(i), std::max(a.storage_at(i), 0.0f));
    }
}

TEST(UnaryOpsTest, ReluProducesCorrectResultOnNonContiguousTensor) {
    // Input tensor a: [[1 0 3], [4 5 -6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    a.storage_at(1) = 0;
    a.storage_at(5) = -6;

    // transpose a: [[1 4], [0 5], [3 -6]]
    a = a.transpose(0, 1);
    mt::Tensor b = a.relu();

    // shape
    EXPECT_EQ(b.shape(), std::vector<size_t>({3, 2}));

    // value
    EXPECT_FLOAT_EQ(b.storage_at(0), 1.0f);
    EXPECT_FLOAT_EQ(b.storage_at(1), 4.0f);
    EXPECT_FLOAT_EQ(b.storage_at(2), 0.0f);
    EXPECT_FLOAT_EQ(b.storage_at(3), 5.0f);
    EXPECT_FLOAT_EQ(b.storage_at(4), 3.0f);
    EXPECT_FLOAT_EQ(b.storage_at(5), 0.0f);
}

TEST(UnaryOpsTest, ExpProducesCorrectResult) {
    // Input tensor a: [[1 0 3], [4 5 -6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    a.storage_at(1) = 0;
    a.storage_at(5) = -6;
    mt::Tensor b = a.exp();

    // shape
    EXPECT_EQ(b.shape(), std::vector<size_t>({2, 3}));

    // value
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(b.storage_at(i), std::exp(a.storage_at(i)));
    }
}

TEST(UnaryOpsTest, ExpProducesCorrectResultOnNonContiguousTensor) {
    // Input tensor a: [[1 0 3], [4 5 -6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    a.storage_at(1) = 0;
    a.storage_at(5) = -6;

    // transpose a: [[1 4], [0 5], [3 -6]]
    a = a.transpose(0, 1);
    mt::Tensor b = a.exp();

    // shape
    EXPECT_EQ(b.shape(), std::vector<size_t>({3, 2}));

    // value
    EXPECT_FLOAT_EQ(b.storage_at(0), std::exp(1.0f));
    EXPECT_FLOAT_EQ(b.storage_at(1), std::exp(4.0f));
    EXPECT_FLOAT_EQ(b.storage_at(2), std::exp(0.0f));
    EXPECT_FLOAT_EQ(b.storage_at(3), std::exp(5.0f));
    EXPECT_FLOAT_EQ(b.storage_at(4), std::exp(3.0f));
    EXPECT_FLOAT_EQ(b.storage_at(5), std::exp(-6.0f));
}

TEST(UnaryOpsTest, LogProducesCorrectResult) {
    // Input tensor a: [[1 0 3], [4 5 -6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    mt::Tensor b = a.log();

    // shape
    EXPECT_EQ(b.shape(), std::vector<size_t>({2, 3}));

    // value
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(b.storage_at(i), std::log(a.storage_at(i)));
    }
}

TEST(UnaryOpsTest, LogProducesCorrectResultOnNonContiguousTensor) {
    // Input tensor a: [[1 0 3], [4 5 -6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    a.storage_at(1) = 0;
    a.storage_at(5) = -6;

    // transpose a: [[1 4], [0 5], [3 -6]]
    a = a.transpose(0, 1);
    mt::Tensor b = a.log();

    // shape
    EXPECT_EQ(b.shape(), std::vector<size_t>({3, 2}));

    // value
    EXPECT_FLOAT_EQ(b.storage_at(0), std::log(1.0f));
    EXPECT_FLOAT_EQ(b.storage_at(1), std::log(4.0f));

    // log(0) -> -inf
    EXPECT_TRUE(std::isinf(b.storage_at(2)));

    EXPECT_FLOAT_EQ(b.storage_at(3), std::log(5.0f));
    EXPECT_FLOAT_EQ(b.storage_at(4), std::log(3.0f));

    // log(-6) -> NaN
    EXPECT_TRUE(std::isnan(b.storage_at(5)));
}