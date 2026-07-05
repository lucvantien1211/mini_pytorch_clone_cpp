#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

TEST(IndexingTest, MutableLinearIndexingAllowsModification) {
    mt::Tensor t({2, 3});

    t.at(0) = 5.0f;
    t.at(1) = 10.0f;

    EXPECT_FLOAT_EQ(t.at(0), 5.0f);
    EXPECT_FLOAT_EQ(t.at(1), 10.0f);
}

TEST(IndexingTest, ConstLinearIndexingAllowsReadOnlyAccess) {
    mt::Tensor t({2, 3});

    t.at(0) = 7.0f;

    const mt::Tensor& ct = t;

    EXPECT_FLOAT_EQ(ct.at(0), 7.0f);
}

TEST(IndexingTest, LinearAtThrowsWhenIndexOutOfRange) {
    mt::Tensor t({2, 3});

    EXPECT_THROW(t.at(100), std::out_of_range);
}

TEST(IndexingTest, MultiDimensionalIndexingReadsCorrectValues) {
    mt::Tensor t = mt::arange(1.0f, 7.0f).reshape({2, 3});

    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({0, 2}), 3.0f);

    EXPECT_FLOAT_EQ(t.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(t.at({1, 2}), 6.0f);
}

TEST(IndexingTest, MultiDimensionalIndexingAllowsModification) {
    mt::Tensor t({2, 3});

    t.at({1, 2}) = 99.0f;

    EXPECT_FLOAT_EQ(t.at({1, 2}), 99.0f);
    EXPECT_FLOAT_EQ(t.at(5), 99.0f);
}

TEST(IndexingTest, MultiDimensionalIndexingThrowsForWrongRank) {
    mt::Tensor t({2, 3});

    EXPECT_THROW((t.at(std::vector<size_t>{1})), std::invalid_argument);

    EXPECT_THROW(t.at({1, 2, 3}), std::invalid_argument);
}

TEST(IndexingTest, MultiDimensionalIndexingThrowsOutOfRangeIndices) {
    mt::Tensor t({2, 3});

    EXPECT_THROW(t.at({2, 0}), std::out_of_range);

    EXPECT_THROW(t.at({0, 3}), std::out_of_range);
}