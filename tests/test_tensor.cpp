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

TEST(TensorTest, ReshapeChangesShape) {
    mt::Tensor t({2, 3});

    mt::Tensor r = t.reshape({3, 2});

    EXPECT_EQ(r.shape()[0], 3);
    EXPECT_EQ(r.shape()[1], 2);
}

TEST(TensorTest, ReshapeDoesNotModifyOriginalTensor) {
    mt::Tensor t({2, 3});

    mt::Tensor r = t.reshape({3, 2});

    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(TensorTest, ReshapePreservesData) {
    mt::Tensor t({2, 3});

    for (size_t i = 0; i < 6; ++i) {
        t.at(i) = static_cast<float>(i);
    }

    mt::Tensor r = t.reshape({3, 2});

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(r.at(i), static_cast<float>(i));
    }
}

TEST(TensorTest, ReshapeThrowsOnInvalidShape) {
    mt::Tensor t({2, 3});

    EXPECT_THROW(t.reshape({4, 2}), std::invalid_argument);
}

TEST(TensorTest, FlattenProducesOneDimensionalTensor) {
    mt::Tensor t({2, 3, 4});

    mt::Tensor f = t.flatten();

    EXPECT_EQ(f.shape().size(), 1);
    EXPECT_EQ(f.shape()[0], 24);
}

TEST(TensorTest, FlattenPreservesData) {
    mt::Tensor t({2, 3});

    for (size_t i = 0; i < 6; i++) {
        t.at(i) = static_cast<float>(i);
    }

    mt::Tensor f = t.flatten();

    for (size_t i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(f.at(i), static_cast<float>(i));
    }
}

TEST(TensorTest, TensorAdditionProducesCorrectResult) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    for (size_t i = 0; i < 6; i++) {
        a.at(i) = static_cast<float>(i + 1);
        b.at(i) = static_cast<float>((i + 1) * 10);
    }

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

TEST(TensorTest, TensorAdditionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    a.at(0) = 1.0f;
    b.at(0) = 10.0f;

    mt::Tensor c = a + b;

    EXPECT_FLOAT_EQ(a.at(0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(0), 11.0f);
}

TEST(TensorTest, TensorAdditionThrowsOnShapeMismatch) {
    mt::Tensor a({2, 3});
    mt::Tensor b({3, 2});

    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(TensorTest, TensorSubtractionProducesCorrectResult) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    for (size_t i = 0; i < 6; i++) {
        a.at(i) = static_cast<float>(i + 1);
        b.at(i) = static_cast<float>((i + 1) * 10);
    }

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

TEST(TensorTest, TensorSubtractionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    a.at(0) = 1.0f;
    b.at(0) = 10.0f;

    mt::Tensor c = a - b;

    EXPECT_FLOAT_EQ(a.at(0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(0), -9.0f);
}

TEST(TensorTest, TensorSubtractionThrowsOnShapeMismatch) {
    mt::Tensor a({2, 3});
    mt::Tensor b({3, 2});

    EXPECT_THROW(a - b, std::invalid_argument);
}

TEST(TensorTest, TensorProductionProducesCorrectResult) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    for (size_t i = 0; i < 6; i++) {
        a.at(i) = static_cast<float>(i + 1);
        b.at(i) = static_cast<float>((i + 1) * 10);
    }

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

TEST(TensorTest, TensorProductionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    a.at(0) = 1.0f;
    b.at(0) = 10.0f;

    mt::Tensor c = a * b;

    EXPECT_FLOAT_EQ(a.at(0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(0), 10.0f);
}

TEST(TensorTest, TensorProductionThrowsOnShapeMismatch) {
    mt::Tensor a({2, 3});
    mt::Tensor b({3, 2});

    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(TensorTest, TensorDivisionProducesCorrectResult) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    for (size_t i = 0; i < 6; i++) {
        a.at(i) = static_cast<float>(i + 1);
        b.at(i) = static_cast<float>((i + 1) * 10);
    }

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

TEST(TensorTest, TensorDivisionDoesNotModifyInputs) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    a.at(0) = 1.0f;
    b.at(0) = 10.0f;

    for (size_t i = 1; i < 6; i++) {
        a.at(i) = 1.0f;
        b.at(i) = 1.0f;
    }

    mt::Tensor c = b / a;

    EXPECT_FLOAT_EQ(a.at(0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(0), 10.0f);
}

TEST(TensorTest, TensorDivisionThrowOnZeroDivisor) {
    mt::Tensor a({2, 3});
    mt::Tensor b({2, 3});

    a.at(0) = 1.0f;
    b.at(0) = 10.0f;

    EXPECT_THROW(a / b, std::runtime_error);
}

TEST(TensorTest, TensorDivisionThrowsOnShapeMismatch) {
    mt::Tensor a({2, 3});
    mt::Tensor b({3, 2});

    EXPECT_THROW(a / b, std::invalid_argument);
}