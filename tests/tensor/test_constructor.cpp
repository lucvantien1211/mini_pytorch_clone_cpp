#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/tensor.h"

TEST(ConstructorTest, DefaultConstructorCreateEmptyTensor) {
    mt::Tensor t;
    EXPECT_EQ(t.shape().size(), 0);
    EXPECT_EQ(t.stride().size(), 0);
    EXPECT_EQ(t.numel(), 0);
}

TEST(ConstructorTest, ConstructorStoresShapeCorrectly) {
    mt::Tensor t({2, 3});

    EXPECT_EQ(t.shape(), std::vector<size_t>({2, 3}));
}

TEST(ConstructorTest, NumelIsComputedCorrectly) {
    mt::Tensor t({2, 3, 4});

    EXPECT_EQ(t.numel(), 24);
}

TEST(ConstructorTest, TensorIsZeroInitialized) {
    mt::Tensor t({2, 3});

    for (size_t i = 0; i < t.numel(); i++) {
        EXPECT_FLOAT_EQ(t.storage_at(i), 0.0f);
    }
}

TEST(ConstructorTest, ConstructorThrowsOnEmptyShape) {
    EXPECT_THROW(mt::Tensor t(std::vector<size_t>{}), std::invalid_argument);
}