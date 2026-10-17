#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "mini_torch/autograd_nodes.h"
#include "mini_torch/factory.h"
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

TEST(AutoGradTest, BackwardRequiresGrad) {
    mt::Tensor x({1});

    EXPECT_THROW(x.backward(), std::runtime_error);
}

TEST(AutoGradTest, BackwardRequiresSingleElement) {
    mt::Tensor x({2}, true);

    EXPECT_THROW(x.backward(), std::runtime_error);
}

TEST(AddBackwardTest, PropagatesGradientToBothInputs) {
    mt::Tensor x({1}, true);
    mt::Tensor y({1}, true);

    x.storage_at(0) = 2.0f;
    y.storage_at(0) = 3.0f;

    mt::Tensor z = x + y;
    z.backward();

    ASSERT_TRUE(x.has_grad());
    ASSERT_TRUE(y.has_grad());

    EXPECT_FLOAT_EQ(x.grad().item(), 1.0f);
    EXPECT_FLOAT_EQ(y.grad().item(), 1.0f);
}

TEST(AddBackwardTest, PropagatesOnlyToInputsRequiringGrad) {
    mt::Tensor x({1}, true);
    mt::Tensor y({1}, false);

    x.storage_at(0) = 2.0f;
    y.storage_at(0) = 3.0f;

    mt::Tensor z = x + y;
    z.backward();

    ASSERT_TRUE(x.has_grad());
    EXPECT_FLOAT_EQ(x.grad().item(), 1.0f);

    EXPECT_FALSE(y.has_grad());
}

TEST(AddBackwardTest, AccumulatesGradientFromMultiplePaths) {
    mt::Tensor x({1}, true);
    mt::Tensor y({1}, true);

    x.storage_at(0) = 2.0f;
    y.storage_at(0) = 3.0f;

    mt::Tensor a = x + y;
    mt::Tensor b = x + y;
    mt::Tensor loss = a + b;

    loss.backward();

    ASSERT_TRUE(x.has_grad());
    ASSERT_TRUE(y.has_grad());

    EXPECT_FLOAT_EQ(x.grad().item(), 2.0f);
    EXPECT_FLOAT_EQ(y.grad().item(), 2.0f);
}

TEST(ReduceToShapeTest, SameShape) {
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    mt::Tensor out = mt::reduce_to_shape(grad, {2, 3});

    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 3}));

    EXPECT_FLOAT_EQ(out.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(out.at({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(out.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(out.at({1, 2}), 6.0f);
}

TEST(ReduceToShapeTest, ReduceLastDimension) {
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    mt::Tensor out = mt::reduce_to_shape(grad, {2, 1});

    EXPECT_EQ(out.shape(), (std::vector<size_t>{2, 1}));

    EXPECT_FLOAT_EQ(out.at({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(out.at({1, 0}), 15.0f);
}

TEST(ReduceToShapeTest, ReduceFirstDimension) {
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    mt::Tensor out = mt::reduce_to_shape(grad, {1, 3});

    EXPECT_EQ(out.shape(), (std::vector<size_t>{1, 3}));

    EXPECT_FLOAT_EQ(out.at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(out.at({0, 1}), 7.0f);
    EXPECT_FLOAT_EQ(out.at({0, 2}), 9.0f);
}

TEST(ReduceToShapeTest, ReduceLeadingDimension) {
    mt::Tensor grad = mt::arange(1.0f, 25.0f).reshape({2, 3, 4});

    mt::Tensor out = mt::reduce_to_shape(grad, {3, 4});

    EXPECT_EQ(out.shape(), (std::vector<size_t>{3, 4}));

    // grad[0, 0, 0] + grad[1, 0, 0] = 1 + 13
    EXPECT_FLOAT_EQ(out.at({0, 0}), 14.0f);

    // grad[0, 2, 3] + grad[1, 2, 3] = 12 + 24
    EXPECT_FLOAT_EQ(out.at({2, 3}), 36.0f);
}

TEST(ReduceToShapeTest, ThrowsForIncompatibleShape) {
    mt::Tensor grad = mt::ones({2, 3});

    EXPECT_THROW(mt::reduce_to_shape(grad, {2, 4}), std::runtime_error);
}

TEST(AddBackwardTest, PropagatesGradOnBroadcastAddition) {
    mt::Tensor a = mt::ones({2, 1}, true);
    mt::Tensor b = mt::ones({1, 3}, true);
    mt::Tensor c = a + b;

    mt::Tensor grad_output = mt::arange(1.0f, 7.0f).reshape({2, 3});

    c.backward_propagate(grad_output);

    EXPECT_EQ(a.grad().shape(), (std::vector<size_t>{2, 1}));

    EXPECT_FLOAT_EQ(a.grad().at({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1, 0}), 15.0f);

    EXPECT_EQ(b.grad().shape(), (std::vector<size_t>{1, 3}));

    EXPECT_FLOAT_EQ(b.grad().at({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(b.grad().at({0, 1}), 7.0f);
    EXPECT_FLOAT_EQ(b.grad().at({0, 2}), 9.0f);
}

TEST(AddBackwardTest, PropagatesGradOnNonBroadcastAddition) {
    mt::Tensor a = mt::ones({2, 2}, true);
    mt::Tensor b = mt::uniform({2, 2}, -5, 5, true);
    mt::Tensor c = a + b;

    mt::Tensor grad_output = mt::arange(1.0f, 5.0f).reshape({2, 2});

    c.backward_propagate(grad_output);

    EXPECT_EQ(a.grad().shape(), (std::vector<size_t>{2, 2}));

    EXPECT_FLOAT_EQ(a.grad().at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad().at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1, 1}), 4.0f);

    EXPECT_EQ(b.grad().shape(), (std::vector<size_t>{2, 2}));

    EXPECT_FLOAT_EQ(b.grad().at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(b.grad().at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(b.grad().at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(b.grad().at({1, 1}), 4.0f);
}

TEST(AddBackwardTest, BroadcastPropagatesToInputsRequiringGrad) {
    mt::Tensor a = mt::ones({2, 1}, true);
    mt::Tensor b = mt::ones({1, 3}, false);
    mt::Tensor c = a + b;

    mt::Tensor grad_output = mt::arange(1.0f, 7.0f).reshape({2, 3});

    c.backward_propagate(grad_output);

    EXPECT_EQ(a.grad().shape(), (std::vector<size_t>{2, 1}));

    EXPECT_FLOAT_EQ(a.grad().at({0, 0}), 6.0f);
    EXPECT_FLOAT_EQ(a.grad().at({1, 0}), 15.0f);

    EXPECT_FALSE(b.has_grad());
}