#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

TEST(ScalarOpsTest, TensorScalarAdditionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    float b = 10.0f;

    mt::Tensor c = a + b;
    mt::Tensor d = b + a;

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(d.shape(), std::vector<size_t>({2, 3}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), 11.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 12.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 13.0f);
    EXPECT_FLOAT_EQ(c.storage_at(3), 14.0f);
    EXPECT_FLOAT_EQ(c.storage_at(4), 15.0f);
    EXPECT_FLOAT_EQ(c.storage_at(5), 16.0f);

    EXPECT_FLOAT_EQ(d.storage_at(0), 11.0f);
    EXPECT_FLOAT_EQ(d.storage_at(1), 12.0f);
    EXPECT_FLOAT_EQ(d.storage_at(2), 13.0f);
    EXPECT_FLOAT_EQ(d.storage_at(3), 14.0f);
    EXPECT_FLOAT_EQ(d.storage_at(4), 15.0f);
    EXPECT_FLOAT_EQ(d.storage_at(5), 16.0f);
}

TEST(ScalarOpsTest, TensorScalarAdditionProducesCorrectResultOnNonContiguous) {
    // Inout tensor a: [[1 4], [2 5], [3 6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3}).transpose(0, 1);
    float b = 10.0f;

    mt::Tensor c = a + b;
    mt::Tensor d = b + a;

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(d.shape(), std::vector<size_t>({3, 2}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), 11.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 14.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 12.0f);
    EXPECT_FLOAT_EQ(c.storage_at(3), 15.0f);
    EXPECT_FLOAT_EQ(c.storage_at(4), 13.0f);
    EXPECT_FLOAT_EQ(c.storage_at(5), 16.0f);

    EXPECT_FLOAT_EQ(d.storage_at(0), 11.0f);
    EXPECT_FLOAT_EQ(d.storage_at(1), 14.0f);
    EXPECT_FLOAT_EQ(d.storage_at(2), 12.0f);
    EXPECT_FLOAT_EQ(d.storage_at(3), 15.0f);
    EXPECT_FLOAT_EQ(d.storage_at(4), 13.0f);
    EXPECT_FLOAT_EQ(d.storage_at(5), 16.0f);
}

TEST(ScalarOpsTest, TensorScalarAdditionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    float b = 10.0f;

    a.storage_at(0) = 1.0f;

    mt::Tensor c = a + b;

    EXPECT_FLOAT_EQ(a.storage_at(0), 1.0f);
    EXPECT_FLOAT_EQ(b, 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(0), 11.0f);
}

TEST(ScalarOpsTest, TensorScalarSubtractionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    float b = 10.0f;

    mt::Tensor c = a - b;
    mt::Tensor d = b - a;

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(d.shape(), std::vector<size_t>({2, 3}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), -9.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), -8.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), -7.0f);
    EXPECT_FLOAT_EQ(c.storage_at(3), -6.0f);
    EXPECT_FLOAT_EQ(c.storage_at(4), -5.0f);
    EXPECT_FLOAT_EQ(c.storage_at(5), -4.0f);

    EXPECT_FLOAT_EQ(d.storage_at(0), 9.0f);
    EXPECT_FLOAT_EQ(d.storage_at(1), 8.0f);
    EXPECT_FLOAT_EQ(d.storage_at(2), 7.0f);
    EXPECT_FLOAT_EQ(d.storage_at(3), 6.0f);
    EXPECT_FLOAT_EQ(d.storage_at(4), 5.0f);
    EXPECT_FLOAT_EQ(d.storage_at(5), 4.0f);
}

TEST(ScalarOpsTest, TensorScalarSubtractionProducesCorrectResultOnNonContiguous) {
    // Inout tensor a: [[1 4], [2 5], [3 6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3}).transpose(0, 1);
    float b = 10.0f;

    mt::Tensor c = a - b;
    mt::Tensor d = b - a;

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(d.shape(), std::vector<size_t>({3, 2}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), -9.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), -6.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), -8.0f);
    EXPECT_FLOAT_EQ(c.storage_at(3), -5.0f);
    EXPECT_FLOAT_EQ(c.storage_at(4), -7.0f);
    EXPECT_FLOAT_EQ(c.storage_at(5), -4.0f);

    EXPECT_FLOAT_EQ(d.storage_at(0), 9.0f);
    EXPECT_FLOAT_EQ(d.storage_at(1), 6.0f);
    EXPECT_FLOAT_EQ(d.storage_at(2), 8.0f);
    EXPECT_FLOAT_EQ(d.storage_at(3), 5.0f);
    EXPECT_FLOAT_EQ(d.storage_at(4), 7.0f);
    EXPECT_FLOAT_EQ(d.storage_at(5), 4.0f);
}

TEST(ScalarOpsTest, TensorScalarSubtractionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    float b = 10.0f;

    a.storage_at(0) = 1.0f;

    mt::Tensor c = a - b;

    EXPECT_FLOAT_EQ(a.storage_at(0), 1.0f);
    EXPECT_FLOAT_EQ(b, 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(0), -9.0f);
}

TEST(ScalarOpsTest, TensorScalarProductionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    float b = 10.0f;

    mt::Tensor c = a * b;
    mt::Tensor d = b * a;

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(d.shape(), std::vector<size_t>({2, 3}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 20.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 30.0f);
    EXPECT_FLOAT_EQ(c.storage_at(3), 40.0f);
    EXPECT_FLOAT_EQ(c.storage_at(4), 50.0f);
    EXPECT_FLOAT_EQ(c.storage_at(5), 60.0f);

    EXPECT_FLOAT_EQ(d.storage_at(0), 10.0f);
    EXPECT_FLOAT_EQ(d.storage_at(1), 20.0f);
    EXPECT_FLOAT_EQ(d.storage_at(2), 30.0f);
    EXPECT_FLOAT_EQ(d.storage_at(3), 40.0f);
    EXPECT_FLOAT_EQ(d.storage_at(4), 50.0f);
    EXPECT_FLOAT_EQ(d.storage_at(5), 60.0f);
}

TEST(ScalarOpsTest, TensorScalarProductionProducesCorrectResultOnNonContiguous) {
    // Inout tensor a: [[1 4], [2 5], [3 6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3}).transpose(0, 1);
    float b = 10.0f;

    mt::Tensor c = a * b;
    mt::Tensor d = b * a;

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(d.shape(), std::vector<size_t>({3, 2}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 40.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 20.0f);
    EXPECT_FLOAT_EQ(c.storage_at(3), 50.0f);
    EXPECT_FLOAT_EQ(c.storage_at(4), 30.0f);
    EXPECT_FLOAT_EQ(c.storage_at(5), 60.0f);

    EXPECT_FLOAT_EQ(d.storage_at(0), 10.0f);
    EXPECT_FLOAT_EQ(d.storage_at(1), 40.0f);
    EXPECT_FLOAT_EQ(d.storage_at(2), 20.0f);
    EXPECT_FLOAT_EQ(d.storage_at(3), 50.0f);
    EXPECT_FLOAT_EQ(d.storage_at(4), 30.0f);
    EXPECT_FLOAT_EQ(d.storage_at(5), 60.0f);
}

TEST(ScalarOpsTest, TensorScalarProductionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    float b = 10.0f;

    a.storage_at(0) = 1.0f;

    mt::Tensor c = a * b;

    EXPECT_FLOAT_EQ(a.storage_at(0), 1.0f);
    EXPECT_FLOAT_EQ(b, 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(0), 10.0f);
}

TEST(ScalarOpsTest, TensorScalarDivisionProducesCorrectResult) {
    mt::Tensor a = mt::ones({2, 3});
    float b = 10.0f;

    mt::Tensor c = b / a;
    mt::Tensor d = a / b;

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));
    EXPECT_EQ(d.shape(), std::vector<size_t>({2, 3}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(3), 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(4), 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(5), 10.0f);

    EXPECT_FLOAT_EQ(d.storage_at(0), 0.1f);
    EXPECT_FLOAT_EQ(d.storage_at(1), 0.1f);
    EXPECT_FLOAT_EQ(d.storage_at(2), 0.1f);
    EXPECT_FLOAT_EQ(d.storage_at(3), 0.1f);
    EXPECT_FLOAT_EQ(d.storage_at(4), 0.1f);
    EXPECT_FLOAT_EQ(d.storage_at(5), 0.1f);
}

TEST(ScalarOpsTest, TensorScalarDivisionProducesCorrectResultOnNonContiguous) {
    // Inout tensor a: [[1 4], [2 5], [3 6]]
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3}).transpose(0, 1);
    float b = 10.0f;

    mt::Tensor c = a / b;
    mt::Tensor d = b / a;

    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({3, 2}));
    EXPECT_EQ(d.shape(), std::vector<size_t>({3, 2}));

    // value
    EXPECT_FLOAT_EQ(c.storage_at(0), 0.1f);
    EXPECT_FLOAT_EQ(c.storage_at(1), 0.4f);
    EXPECT_FLOAT_EQ(c.storage_at(2), 0.2f);
    EXPECT_FLOAT_EQ(c.storage_at(3), 0.5f);
    EXPECT_FLOAT_EQ(c.storage_at(4), 0.3f);
    EXPECT_FLOAT_EQ(c.storage_at(5), 0.6f);

    EXPECT_FLOAT_EQ(d.storage_at(0), 10.0f);
    EXPECT_FLOAT_EQ(d.storage_at(1), 2.5f);
    EXPECT_FLOAT_EQ(d.storage_at(2), 5.0f);
    EXPECT_FLOAT_EQ(d.storage_at(3), 2.0f);
    EXPECT_FLOAT_EQ(d.storage_at(4), 10.0f / 3.0f);
    EXPECT_FLOAT_EQ(d.storage_at(5), 10.0f / 6.0f);
}

TEST(ScalarOpsTest, TensorScalarDivisionDoesNotModifyInputs) {
    mt::Tensor a = mt::ones({2, 3});
    float b = 10.0f;

    mt::Tensor c = b / a;

    EXPECT_FLOAT_EQ(a.storage_at(0), 1.0f);
    EXPECT_FLOAT_EQ(b, 10.0f);
    EXPECT_FLOAT_EQ(c.storage_at(0), 10.0f);
}

TEST(ScalarOpsTest, TensorScalarDivisionThrowOnZeroDivisor) {
    mt::Tensor a({2, 3});
    float b = 10.0f;
    float c = 0.0;

    EXPECT_THROW(b / a, std::runtime_error);
    EXPECT_THROW(a / c, std::runtime_error);
}