#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

TEST(ViewTest, ReshapeChangesShape) {
    mt::Tensor t({2, 3});

    mt::Tensor r = t.reshape({3, 2});

    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 2);
}

TEST(ViewTest, ReshapeDoesNotModifyOriginalTensor) {
    mt::Tensor t({2, 3});

    mt::Tensor r = t.reshape({3, 2});

    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(ViewTest, ReshapePreservesData) {
    mt::Tensor t = mt::arange(0.0f, 6.0f).reshape({2, 3});

    mt::Tensor r = t.reshape({3, 2});

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(r.at(i), static_cast<float>(i));
    }
}

TEST(ViewTest, ReshapeThrowsOnInvalidShape) {
    mt::Tensor t({2, 3});

    EXPECT_THROW(t.reshape({4, 2}), std::invalid_argument);
}

TEST(ViewTest, FlattenProducesOneDimensionalTensor) {
    mt::Tensor t({2, 3, 4});

    mt::Tensor f = t.flatten();

    EXPECT_EQ(f.shape().size(), 1);
    EXPECT_EQ(f.shape()[0], 24);
}

TEST(ViewTest, FlattenPreservesData) {
    mt::Tensor t = mt::arange(0.0f, 6.0f).reshape({2, 3});

    mt::Tensor f = t.flatten();

    for (size_t i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(f.at(i), static_cast<float>(i));
    }
}