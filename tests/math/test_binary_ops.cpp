#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

TEST(BinaryOpsTest, TensorAdditionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    mt::Tensor b = 10 * a;

    mt::Tensor c = a + b;
    mt::Tensor d = b + a;

    // shape
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    EXPECT_EQ(d.shape()[0], 2);
    EXPECT_EQ(d.shape()[1], 3);

    // value
    EXPECT_FLOAT_EQ(c.at(0), 11.0f);
    EXPECT_FLOAT_EQ(c.at(1), 22.0f);
    EXPECT_FLOAT_EQ(c.at(2), 33.0f);
    EXPECT_FLOAT_EQ(c.at(3), 44.0f);
    EXPECT_FLOAT_EQ(c.at(4), 55.0f);
    EXPECT_FLOAT_EQ(c.at(5), 66.0f);

    EXPECT_FLOAT_EQ(d.at(0), 11.0f);
    EXPECT_FLOAT_EQ(d.at(1), 22.0f);
    EXPECT_FLOAT_EQ(d.at(2), 33.0f);
    EXPECT_FLOAT_EQ(d.at(3), 44.0f);
    EXPECT_FLOAT_EQ(d.at(4), 55.0f);
    EXPECT_FLOAT_EQ(d.at(5), 66.0f);
}

TEST(BinaryOpsTest, TensorAdditionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    a.at(0) = 1.0f;
    b.at(0) = 10.0f;

    mt::Tensor c = a + b;

    EXPECT_FLOAT_EQ(a.at(0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(0), 11.0f);
}

TEST(BinaryOpsTest, TensorAdditionThrowsOnShapeMismatch) {
    mt::Tensor a({2, 3});
    mt::Tensor b({3, 2});

    EXPECT_THROW(a + b, std::runtime_error);
}

TEST(BinaryOpsTest, TensorSubtractionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    mt::Tensor b = 10 * a;

    mt::Tensor c = a - b;
    mt::Tensor d = b - a;

    // shape
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    EXPECT_EQ(d.shape()[0], 2);
    EXPECT_EQ(d.shape()[1], 3);

    // value
    EXPECT_FLOAT_EQ(c.at(0), -9.0f);
    EXPECT_FLOAT_EQ(c.at(1), -18.0f);
    EXPECT_FLOAT_EQ(c.at(2), -27.0f);
    EXPECT_FLOAT_EQ(c.at(3), -36.0f);
    EXPECT_FLOAT_EQ(c.at(4), -45.0f);
    EXPECT_FLOAT_EQ(c.at(5), -54.0f);

    EXPECT_FLOAT_EQ(d.at(0), 9.0f);
    EXPECT_FLOAT_EQ(d.at(1), 18.0f);
    EXPECT_FLOAT_EQ(d.at(2), 27.0f);
    EXPECT_FLOAT_EQ(d.at(3), 36.0f);
    EXPECT_FLOAT_EQ(d.at(4), 45.0f);
    EXPECT_FLOAT_EQ(d.at(5), 54.0f);
}

TEST(BinaryOpsTest, TensorSubtractionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    a.at(0) = 1.0f;
    b.at(0) = 10.0f;

    mt::Tensor c = a - b;

    EXPECT_FLOAT_EQ(a.at(0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(0), -9.0f);
}

TEST(BinaryOpsTest, TensorSubtractionThrowsOnShapeMismatch) {
    mt::Tensor a({2, 3});
    mt::Tensor b({3, 2});

    EXPECT_THROW(a - b, std::invalid_argument);
}

TEST(BinaryOpsTest, TensorProductionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    mt::Tensor b = 10 * a;

    mt::Tensor c = a * b;
    mt::Tensor d = b * a;

    // shape
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    EXPECT_EQ(d.shape()[0], 2);
    EXPECT_EQ(d.shape()[1], 3);

    // value
    EXPECT_FLOAT_EQ(c.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(1), 40.0f);
    EXPECT_FLOAT_EQ(c.at(2), 90.0f);
    EXPECT_FLOAT_EQ(c.at(3), 160.0f);
    EXPECT_FLOAT_EQ(c.at(4), 250.0f);
    EXPECT_FLOAT_EQ(c.at(5), 360.0f);

    EXPECT_FLOAT_EQ(d.at(0), 10.0f);
    EXPECT_FLOAT_EQ(d.at(1), 40.0f);
    EXPECT_FLOAT_EQ(d.at(2), 90.0f);
    EXPECT_FLOAT_EQ(d.at(3), 160.0f);
    EXPECT_FLOAT_EQ(d.at(4), 250.0f);
    EXPECT_FLOAT_EQ(d.at(5), 360.0f);
}

TEST(BinaryOpsTest, TensorProductionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    a.at(0) = 1.0f;
    b.at(0) = 10.0f;

    mt::Tensor c = a * b;

    EXPECT_FLOAT_EQ(a.at(0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(0), 10.0f);
}

TEST(BinaryOpsTest, TensorProductionThrowsOnShapeMismatch) {
    mt::Tensor a({2, 3});
    mt::Tensor b({3, 2});

    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(BinaryOpsTest, TensorDivisionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    mt::Tensor b = 10 * a;

    mt::Tensor c = b / a;
    mt::Tensor d = a / b;

    // shape
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    EXPECT_EQ(d.shape()[0], 2);
    EXPECT_EQ(d.shape()[1], 3);

    // value
    EXPECT_FLOAT_EQ(c.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(1), 10.0f);
    EXPECT_FLOAT_EQ(c.at(2), 10.0f);
    EXPECT_FLOAT_EQ(c.at(3), 10.0f);
    EXPECT_FLOAT_EQ(c.at(4), 10.0f);
    EXPECT_FLOAT_EQ(c.at(5), 10.0f);

    EXPECT_FLOAT_EQ(d.at(0), 0.1f);
    EXPECT_FLOAT_EQ(d.at(1), 0.1f);
    EXPECT_FLOAT_EQ(d.at(2), 0.1f);
    EXPECT_FLOAT_EQ(d.at(3), 0.1f);
    EXPECT_FLOAT_EQ(d.at(4), 0.1f);
    EXPECT_FLOAT_EQ(d.at(5), 0.1f);
}

TEST(BinaryOpsTest, TensorDivisionDoesNotModifyInputs) {
    mt::Tensor a = mt::ones({2, 3});
    mt::Tensor b = mt::ones({2, 3});
    b.at(0) = 10.0f;

    mt::Tensor c = b / a;

    EXPECT_FLOAT_EQ(a.at(0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(0), 10.0f);
}

TEST(BinaryOpsTest, TensorDivisionThrowOnZeroDivisor) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    a.at(0) = 1.0f;
    b.at(0) = 10.0f;

    EXPECT_THROW(a / b, std::runtime_error);
}

TEST(BinaryOpsTest, TensorDivisionThrowsOnShapeMismatch) {
    mt::Tensor a({2, 3});
    mt::Tensor b({3, 2});

    EXPECT_THROW(a / b, std::invalid_argument);
}

TEST(BinaryOpsTest, BroadcastAdditionProducesCorrectResult) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    mt::Tensor b = mt::ones({3});

    mt::Tensor c = a + b;
    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));

    // value
    EXPECT_FLOAT_EQ(c.at(0), 2.0f);
    EXPECT_FLOAT_EQ(c.at(1), 3.0f);
    EXPECT_FLOAT_EQ(c.at(2), 4.0f);
    EXPECT_FLOAT_EQ(c.at(3), 5.0f);
    EXPECT_FLOAT_EQ(c.at(4), 6.0f);
    EXPECT_FLOAT_EQ(c.at(5), 7.0f);
}

TEST(BinaryOpsTest, AdditionNonContiguousTensor) {
    mt::Tensor a = mt::arange(1.0f, 7.0f).reshape({2, 3});
    mt::Tensor b = mt::arange(1.0f, 7.0f).reshape({3, 2}).transpose(0, 1);

    mt::Tensor c = a + b;
    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 3}));

    // value
    EXPECT_FLOAT_EQ(c.at(0), 2.0f);
    EXPECT_FLOAT_EQ(c.at(1), 5.0f);
    EXPECT_FLOAT_EQ(c.at(2), 8.0f);
    EXPECT_FLOAT_EQ(c.at(3), 6.0f);
    EXPECT_FLOAT_EQ(c.at(4), 9.0f);
    EXPECT_FLOAT_EQ(c.at(5), 12.0f);
}

TEST(BinaryOpsTest, BroadcastMiddleDim) {
    mt::Tensor a = mt::arange(1.0f, 13.0f).reshape({2, 2, 3});
    mt::Tensor b = mt::ones({2, 1, 3});

    mt::Tensor c = a + b;
    // shape
    EXPECT_EQ(c.shape(), std::vector<size_t>({2, 2, 3}));

    // value
    EXPECT_FLOAT_EQ(c.at(0), 2.0f);
    EXPECT_FLOAT_EQ(c.at(1), 3.0f);
    EXPECT_FLOAT_EQ(c.at(2), 4.0f);
    EXPECT_FLOAT_EQ(c.at(3), 5.0f);
    EXPECT_FLOAT_EQ(c.at(4), 6.0f);
    EXPECT_FLOAT_EQ(c.at(5), 7.0f);
    EXPECT_FLOAT_EQ(c.at(6), 8.0f);
    EXPECT_FLOAT_EQ(c.at(7), 9.0f);
    EXPECT_FLOAT_EQ(c.at(8), 10.0f);
    EXPECT_FLOAT_EQ(c.at(9), 11.0f);
    EXPECT_FLOAT_EQ(c.at(10), 12.0f);
    EXPECT_FLOAT_EQ(c.at(11), 13.0f);
}