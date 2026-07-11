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

TEST(ViewTest, ReshapeSharesStorage) {
    mt::Tensor a({2, 3});

    mt::Tensor b = a.reshape({3, 2});

    b.at(3) = 10.0f;

    EXPECT_FLOAT_EQ(a.at(3), 10.0f);
}

TEST(ViewTest, ReshapeThrowsOnNonContiguousTensor) {
    mt::Tensor a = mt::arange(0.0f, 6.0f).reshape({2, 3}).transpose(0, 1);

    EXPECT_THROW(a.reshape({2, 3}), std::runtime_error);
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

TEST(ViewTest, PermuteChangeDimensionOrderCorrectly) {
    mt::Tensor t({2, 3, 4});  // stride = {12, 4, 1}

    mt::Tensor p = t.permute({1, 2, 0});

    // shape
    EXPECT_EQ(p.shape()[0], 3);
    EXPECT_EQ(p.shape()[1], 4);
    EXPECT_EQ(p.shape()[2], 2);

    // stride
    EXPECT_EQ(p.stride()[0], 4);
    EXPECT_EQ(p.stride()[1], 1);
    EXPECT_EQ(p.stride()[2], 12);

    // contiguous
    EXPECT_FALSE(p.is_contiguous());
}

TEST(ViewTest, IdentityPermutation) {
    mt::Tensor t({2, 3, 4});  // stride = {12, 4, 1}

    mt::Tensor p = t.permute({0, 1, 2});

    // shape
    EXPECT_EQ(p.shape()[0], 2);
    EXPECT_EQ(p.shape()[1], 3);
    EXPECT_EQ(p.shape()[2], 4);

    // stride
    EXPECT_EQ(p.stride()[0], 12);
    EXPECT_EQ(p.stride()[1], 4);
    EXPECT_EQ(p.stride()[2], 1);

    // contiguous
    EXPECT_TRUE(p.is_contiguous());
}

TEST(ViewTest, PermuteSharesStorage) {
    mt::Tensor t({2, 3, 4});
    mt::Tensor p = t.permute({1, 2, 0});
    p.at(3) = 3.0f;

    EXPECT_FLOAT_EQ(t.at(3), 3.0f);
}

TEST(ViewTest, PermuteThrowsOnMismatchNDims) {
    mt::Tensor t({2, 3, 4});
    EXPECT_THROW(t.permute({2, 0}), std::invalid_argument);
}

TEST(ViewTest, PermuteThrowsOnDuplicateDimIndex) {
    mt::Tensor t({2, 3, 4});
    EXPECT_THROW(t.permute({2, 0, 2}), std::invalid_argument);
}

TEST(ViewTest, PermuteThrowsOnOutOfRangeDimIndex) {
    mt::Tensor t({2, 3, 4});
    EXPECT_THROW(t.permute({2, 0, 3}), std::out_of_range);
}

TEST(ViewTest, TransposeChangeDimensionOrderCorrectly) {
    mt::Tensor t({2, 3, 4});  // stride = {12, 4, 1}

    mt::Tensor p = t.transpose(1, 2);

    // shape
    EXPECT_EQ(p.shape()[0], 2);
    EXPECT_EQ(p.shape()[1], 4);
    EXPECT_EQ(p.shape()[2], 3);

    // stride
    EXPECT_EQ(p.stride()[0], 12);
    EXPECT_EQ(p.stride()[1], 1);
    EXPECT_EQ(p.stride()[2], 4);

    // contiguous
    EXPECT_FALSE(p.is_contiguous());
}

TEST(ViewTest, TransposeChangesDataIndexingCorrectly) {
    // Original tensor with shape (2, 3): [[0, 1, 2], [3, 4, 5]]
    mt::Tensor t = mt::arange(0.0f, 6.0f).reshape({2, 3});

    // Transpose (3, 2): [[0, 3], [1, 4], [2, 5]]
    mt::Tensor p = t.transpose(0, 1);

    EXPECT_FLOAT_EQ(p.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(p.at({0, 1}), 3.0f);
    EXPECT_FLOAT_EQ(p.at({1, 0}), 1.0f);
    EXPECT_FLOAT_EQ(p.at({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(p.at({2, 0}), 2.0f);
    EXPECT_FLOAT_EQ(p.at({2, 1}), 5.0f);
}

TEST(ViewTest, RepeatedlyTransposeSameDimReturnsOriginal) {
    mt::Tensor t = mt::arange(0.0f, 6.0f).reshape({2, 3});

    mt::Tensor p = t.transpose(0, 1).transpose(0, 1);

    // shape
    EXPECT_EQ(p.shape()[0], 2);
    EXPECT_EQ(p.shape()[1], 3);

    // stride
    EXPECT_EQ(p.stride()[0], 3);
    EXPECT_EQ(p.stride()[1], 1);

    // contiguous
    EXPECT_TRUE(p.is_contiguous());

    // Data
    EXPECT_FLOAT_EQ(p.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(p.at({0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(p.at({0, 2}), 2.0f);

    EXPECT_FLOAT_EQ(p.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(p.at({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(p.at({1, 2}), 5.0f);
}

TEST(ViewTest, TransposeThrowsOnOutOfRangeDimIndex) {
    mt::Tensor t({2, 3});
    EXPECT_THROW(t.transpose(0, 2), std::out_of_range);
}