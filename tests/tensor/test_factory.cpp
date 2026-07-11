#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

TEST(FactoryTest, ZerosFactoryFunctionProducesCorrectResult) {
    mt::Tensor t = mt::zeros({2, 3});

    // shape
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);

    // value
    for (size_t i = 0; i < t.numel(); i++) {
        EXPECT_FLOAT_EQ(t.at(i), 0.0f);
    }
}

TEST(FactoryTest, OnesFactoryFunctionProducesCorrectResult) {
    mt::Tensor t = mt::ones({2, 3});

    // shape
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);

    // value
    for (size_t i = 0; i < t.numel(); i++) {
        EXPECT_FLOAT_EQ(t.at(i), 1.0f);
    }
}

TEST(FactoryTest, ArangeFactoryFunctionProducesCorrectResult) {
    mt::Tensor t = mt::arange(0.0f, 6.0f);

    // numel
    EXPECT_EQ(t.numel(), 6);

    // shape
    EXPECT_EQ(t.shape().size(), 1);
    EXPECT_EQ(t.shape()[0], 6);

    // value
    for (size_t i = 0; i < t.numel(); i++) {
        EXPECT_FLOAT_EQ(t.at(i), static_cast<float>(i));
    }
}

TEST(FactoryTest, ArangeFactoryFunctionProducesCorrectResultOnNegativeStep) {
    mt::Tensor t = mt::arange(5.0f, 0.0f, -1.0f);

    // numel
    EXPECT_EQ(t.numel(), 5);

    // shape
    EXPECT_EQ(t.shape().size(), 1);
    EXPECT_EQ(t.shape()[0], 5);

    // value
    for (size_t i = 0; i < t.numel(); i++) {
        EXPECT_FLOAT_EQ(t.at(i), 5.0f - static_cast<float>(i));
    }
}

TEST(FactoryTest, ArangeFactoryFunctionThrowsOnZeroStep) {
    EXPECT_THROW(mt::arange(0.0f, 6.0f, 0.0f), std::invalid_argument);
}

TEST(FactoryTest, ArangeFactoryFunctionReturnsEmptyTensor) {
    // start > end, step > 0
    mt::Tensor a = mt::arange(10.0f, 5.0f);

    // start == end
    mt::Tensor b = mt::arange(5.0f, 5.0f);

    EXPECT_EQ(a.numel(), 0);
    EXPECT_EQ(b.numel(), 0);
}