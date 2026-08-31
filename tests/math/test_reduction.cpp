#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

TEST(ReductionTest, SumReductionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0, 7.0).reshape({3, 2});

    mt::Tensor b = a.sum(0);
    mt::Tensor c = a.sum(1);
    float total = a.sum();

    // shape
    EXPECT_EQ(b.shape(), std::vector<size_t>({2}));
    EXPECT_EQ(c.shape(), std::vector<size_t>({3}));

    // value
    EXPECT_FLOAT_EQ(b.storage_at(0), 9.0f);
    EXPECT_FLOAT_EQ(b.storage_at(1), 12.0f);

    EXPECT_FLOAT_EQ(c.storage_at(0), 3.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 7.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 11.0f);

    EXPECT_FLOAT_EQ(total, 21.0f);
}

TEST(ReductionTest, SumReductionOnNonContiguous) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({3, 2});

    mt::Tensor b = a.transpose(0, 1);

    mt::Tensor out = b.sum(0);

    EXPECT_EQ(out.shape(), std::vector<size_t>({3}));

    EXPECT_FLOAT_EQ(out.storage_at(0), 3.0f);
    EXPECT_FLOAT_EQ(out.storage_at(1), 7.0f);
    EXPECT_FLOAT_EQ(out.storage_at(2), 11.0f);
}

TEST(ReductionTest, SumReductionThrowsOnOutOfRangeDim) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({3, 2});

    EXPECT_THROW(a.sum(100), std::out_of_range);
}

TEST(ReductionTest, MeanReductionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0, 7.0).reshape({3, 2});

    mt::Tensor b = a.mean(0);
    mt::Tensor c = a.mean(1);
    float total_mean = a.mean();

    // shape
    EXPECT_EQ(b.shape(), std::vector<size_t>({2}));
    EXPECT_EQ(c.shape(), std::vector<size_t>({3}));

    // value
    EXPECT_FLOAT_EQ(b.storage_at(0), 3.0f);
    EXPECT_FLOAT_EQ(b.storage_at(1), 4.0f);

    EXPECT_FLOAT_EQ(c.storage_at(0), 1.5f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 3.5f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 5.5f);

    EXPECT_FLOAT_EQ(total_mean, 3.5f);
}

TEST(ReductionTest, MeanReductionOnNonContiguous) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({3, 2});

    mt::Tensor b = a.transpose(0, 1);

    mt::Tensor out = b.mean(0);

    EXPECT_EQ(out.shape(), std::vector<size_t>({3}));

    EXPECT_FLOAT_EQ(out.storage_at(0), 1.5f);
    EXPECT_FLOAT_EQ(out.storage_at(1), 3.5f);
    EXPECT_FLOAT_EQ(out.storage_at(2), 5.5f);
}

TEST(ReductionTest, MeanReductionThrowsOnOutOfRangeDim) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({3, 2});

    EXPECT_THROW(a.mean(100), std::out_of_range);
}