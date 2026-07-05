#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/tensor.h"

TEST(ConstructorTest, ConstructorStoresShapeCorrectly) {
    mt::Tensor t({2, 3});

    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(ConstructorTest, NumelIsComputedCorrectly) {
    mt::Tensor t({2, 3, 4});

    EXPECT_EQ(t.numel(), 24);
}

TEST(ConstructorTest, TensorIsZeroInitialized) {
    mt::Tensor t({2, 3});

    for (size_t i = 0; i < t.numel(); i++) {
        EXPECT_FLOAT_EQ(t.at(i), 0.0f);
    }
}