#include <gtest/gtest.h>

#include <stdexcept>

#include "mini_torch/tensor.h"

TEST(TensorTest, ConstructorStoresShapeCorrectly) {
    mt::Tensor t({2, 3});

    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(TensorTest, NumelIsComputedCorrectly) {
    mt::Tensor t({2, 3, 4});

    EXPECT_EQ(t.numel(), 24);
}

TEST(TensorTest, TensorIsZeroInitialized) {
    mt::Tensor t({2, 3});

    for (size_t i = 0; i < t.numel(); i++) {
        EXPECT_FLOAT_EQ(t.at(i), 0.0f);
    }
}

TEST(TensorTest, MutableAtAllowsModification) {
    mt::Tensor t({2, 3});

    t.at(0) = 5.0f;
    t.at(1) = 10.0f;

    EXPECT_FLOAT_EQ(t.at(0), 5.0f);
    EXPECT_FLOAT_EQ(t.at(1), 10.0f);
}

TEST(TensorTest, ConstAtAllowsReadOnlyAccess) {
    mt::Tensor t({2, 3});

    t.at(0) = 7.0f;

    const mt::Tensor& ct = t;

    EXPECT_FLOAT_EQ(ct.at(0), 7.0f);
}

TEST(TensorTest, AtThrowsWhenIndexOutOfRange) {
    mt::Tensor t({2, 3});

    EXPECT_THROW(t.at(100), std::out_of_range);
}