#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

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

TEST(TensorTest, MutableLinearIndexingAllowsModification) {
    mt::Tensor t({2, 3});

    t.at(0) = 5.0f;
    t.at(1) = 10.0f;

    EXPECT_FLOAT_EQ(t.at(0), 5.0f);
    EXPECT_FLOAT_EQ(t.at(1), 10.0f);
}

TEST(TensorTest, ConstLinearIndexingAllowsReadOnlyAccess) {
    mt::Tensor t({2, 3});

    t.at(0) = 7.0f;

    const mt::Tensor& ct = t;

    EXPECT_FLOAT_EQ(ct.at(0), 7.0f);
}

TEST(TensorTest, LinearAtThrowsWhenIndexOutOfRange) {
    mt::Tensor t({2, 3});

    EXPECT_THROW(t.at(100), std::out_of_range);
}

// TEST(TensorTest, StrideIsComputedCorrectly) {
//     mt::Tensor t({2, 3, 4});

//     const auto& stride = t.get_stride();

//     EXPECT_EQ(stride.size(), 3);

//     EXPECT_EQ(stride[0], 12);
//     EXPECT_EQ(stride[1], 4);
//     EXPECT_EQ(stride[2], 1);
// }

// TEST(TensorTest, LinearIndexIsComputedCorrectly) {
//     mt::Tensor t({2, 3});

//     EXPECT_EQ(t.get_linear_index({0, 0}), 0);
//     EXPECT_EQ(t.get_linear_index({0, 1}), 1);
//     EXPECT_EQ(t.get_linear_index({0, 2}), 2);

//     EXPECT_EQ(t.get_linear_index({1, 0}), 3);
//     EXPECT_EQ(t.get_linear_index({1, 1}), 4);
//     EXPECT_EQ(t.get_linear_index({1, 2}), 5);
// }

TEST(TensorTest, MultiDimensionalIndexingReadsCorrectValues) {
    mt::Tensor t({2, 3});

    for (size_t i = 0; i < t.numel(); i++) {
        t.at(i) = static_cast<float>(i + 1);
    }

    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);

    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST(TensorTest, MultiDimensionalIndexingAllowsModification) {
    mt::Tensor t({2, 3});

    t.at({1, 2}) = 99.0f;

    EXPECT_FLOAT_EQ(t.at({1, 2}), 99.0f);
    EXPECT_FLOAT_EQ(t.at(5), 99.0f);
}

// TEST(TensorTest, MultiDimensionalIndexingThrowsForWrongNumberOfIndices) {
//     mt::Tensor t({2, 3});

//     EXPECT_THROW(t.get_linear_index({1}), std::invalid_argument);

//     EXPECT_THROW(t.get_linear_index({1, 2, 3}), std::invalid_argument);
// }

// TEST(TensorTest, MultiDimensionalIndexingThrowsForOutOfRangeIndices) {
//     mt::Tensor t({2, 3});

//     EXPECT_THROW(t.get_linear_index({2, 0}), std::out_of_range);

//     EXPECT_THROW(t.get_linear_index({0, 3}), std::out_of_range);
// }