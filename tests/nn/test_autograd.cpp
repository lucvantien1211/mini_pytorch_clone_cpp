#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/tensor.h"

TEST(AutoGradTest, AddBackwardTest) {
    mt::Tensor a({2}, true);
    mt::Tensor b({2}, true);

    mt::Tensor c = a + b;

    EXPECT_TRUE(c.requires_grad());
    EXPECT_FALSE(c.is_leaf());
    EXPECT_TRUE(c.grad_fn()->name() == "AddBackward");
}

TEST(AutoGradTest, AddWithoutGradDoesNotBuildGraph) {
    mt::Tensor a({2});
    mt::Tensor b({2});

    mt::Tensor c = a + b;

    EXPECT_FALSE(c.requires_grad());
    EXPECT_TRUE(c.is_leaf());
    EXPECT_EQ(c.grad_fn(), nullptr);
}

TEST(AutoGradTest, GradInitiallyAbsent) {
    mt::Tensor x({2}, true);

    EXPECT_TRUE(x.requires_grad());
    EXPECT_FALSE(x.has_grad());
}

TEST(AutoGradTest, NoGradByDefault) {
    mt::Tensor x({2});

    EXPECT_FALSE(x.requires_grad());
    EXPECT_FALSE(x.has_grad());
}

TEST(AutoGradTest, GradThrowsWhenAbsent) {
    mt::Tensor x({2}, true);

    EXPECT_THROW(x.grad(), std::runtime_error);
}

TEST(AutoGradTest, AccumulateGradFirstTime) {
    mt::Tensor x({2}, true);
    mt::Tensor grad({2});

    grad.storage_at(0) = 3.0f;
    grad.storage_at(1) = 5.0f;

    x.accumulate_grad(grad);

    EXPECT_TRUE(x.has_grad());

    EXPECT_FLOAT_EQ(x.grad().storage_at(0), 3.0f);
    EXPECT_FLOAT_EQ(x.grad().storage_at(1), 5.0f);
}

TEST(AutoGradTest, AccumulateGradMultipleTimes) {
    mt::Tensor x({2}, true);

    mt::Tensor grad1({2});
    grad1.storage_at(0) = 3.0f;
    grad1.storage_at(1) = 5.0f;

    mt::Tensor grad2({2});
    grad2.storage_at(0) = 2.0f;
    grad2.storage_at(1) = 4.0f;

    x.accumulate_grad(grad1);
    x.accumulate_grad(grad2);

    EXPECT_FLOAT_EQ(x.grad().storage_at(0), 5.0f);
    EXPECT_FLOAT_EQ(x.grad().storage_at(1), 9.0f);
}

TEST(AutoGradTest, ZeroGrad) {
    mt::Tensor x({2}, true);

    mt::Tensor grad1({2});
    grad1.storage_at(0) = 3.0f;
    grad1.storage_at(1) = 5.0f;

    mt::Tensor grad2({2});
    grad2.storage_at(0) = 2.0f;
    grad2.storage_at(1) = 4.0f;

    x.accumulate_grad(grad1);
    x.accumulate_grad(grad2);  // now gradient should be [5 9]

    x.zero_grad();  // reset

    x.accumulate_grad(grad1);
    EXPECT_FLOAT_EQ(x.grad().storage_at(0), 3.0f);
    EXPECT_FLOAT_EQ(x.grad().storage_at(1), 5.0f);
}

TEST(AutoGradTest, AccumulateGradDoesNotAliasIncoming) {
    mt::Tensor x({2}, true);

    mt::Tensor incoming({2});
    incoming.storage_at(0) = 3.0f;
    incoming.storage_at(1) = 5.0f;

    x.accumulate_grad(incoming);

    incoming.storage_at(0) = 100.0f;
    incoming.storage_at(1) = 200.0f;

    EXPECT_FLOAT_EQ(x.grad().storage_at(0), 3.0f);
    EXPECT_FLOAT_EQ(x.grad().storage_at(1), 5.0f);
}

TEST(AutoGradTest, GradientDoesNotBuildGraph) {
    mt::Tensor x({2}, true);

    mt::Tensor incoming({2});
    incoming.storage_at(0) = 3.0f;
    incoming.storage_at(1) = 5.0f;

    x.accumulate_grad(incoming);

    EXPECT_FALSE(x.grad().requires_grad());
    EXPECT_TRUE(x.grad().is_leaf());
    EXPECT_EQ(x.grad().grad_fn(), nullptr);
}