#include <gtest/gtest.h>

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_torch/autograd.h"
#include "mini_torch/autograd_nodes.h"
#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

namespace {

void set_values(mt::Tensor& tensor, std::initializer_list<float> values) {
    ASSERT_EQ(tensor.numel(), values.size());
    size_t i = 0;
    for (float value : values) {
        tensor.storage_at(i++) = value;
    }
}

void expect_values(mt::Tensor tensor, std::initializer_list<float> expected) {
    ASSERT_EQ(tensor.numel(), expected.size());
    size_t i = 0;
    for (float value : expected) {
        EXPECT_FLOAT_EQ(tensor.storage_at(i++), value);
    }
}

void expect_shape(const mt::Tensor& tensor, const std::vector<size_t>& expected_shape) {
    EXPECT_EQ(tensor.shape(), expected_shape);
}

}  // namespace

// -----------------------------------------------------------------------------
// Tensor autograd metadata and gradient storage
// -----------------------------------------------------------------------------

TEST(AutoGradMetadataTest, OperationBuildsGraphWhenAnInputRequiresGrad) {
    mt::Tensor a({2}, true);
    mt::Tensor b({2}, false);

    mt::Tensor out = a + b;

    EXPECT_TRUE(out.requires_grad());
    EXPECT_FALSE(out.is_leaf());
    ASSERT_NE(out.grad_fn(), nullptr);
    EXPECT_EQ(out.grad_fn()->name(), "AddBackward");
}

TEST(AutoGradMetadataTest, OperationDoesNotBuildGraphWhenNoInputRequiresGrad) {
    mt::Tensor a({2}, false);
    mt::Tensor b({2}, false);

    mt::Tensor out = a + b;

    EXPECT_FALSE(out.requires_grad());
    EXPECT_TRUE(out.is_leaf());
    EXPECT_EQ(out.grad_fn(), nullptr);
}

TEST(AutoGradMetadataTest, TensorWithoutGradTrackingHasNoGradient) {
    mt::Tensor x({2}, false);

    EXPECT_FALSE(x.requires_grad());
    EXPECT_FALSE(x.has_grad());
}

TEST(AutoGradMetadataTest, GradientIsInitiallyAbsent) {
    mt::Tensor x({2}, true);

    EXPECT_TRUE(x.requires_grad());
    EXPECT_FALSE(x.has_grad());
}

TEST(AutoGradMetadataTest, GradThrowsWhenGradientIsAbsent) {
    mt::Tensor x({2}, true);

    EXPECT_THROW(x.grad(), std::runtime_error);
}

TEST(AutoGradStorageTest, AccumulateGradientForTheFirstTime) {
    mt::Tensor x({2}, true);
    mt::Tensor incoming({2});
    set_values(incoming, {3.0f, 5.0f});

    x.accumulate_grad(incoming);

    ASSERT_TRUE(x.has_grad());
    expect_values(x.grad(), {3.0f, 5.0f});
}

TEST(AutoGradStorageTest, AccumulateGradientMultipleTimes) {
    mt::Tensor x({2}, true);
    mt::Tensor grad1({2});
    mt::Tensor grad2({2});
    set_values(grad1, {3.0f, 5.0f});
    set_values(grad2, {2.0f, 4.0f});

    x.accumulate_grad(grad1);
    x.accumulate_grad(grad2);

    expect_values(x.grad(), {5.0f, 9.0f});
}

TEST(AutoGradStorageTest, ZeroGradResetsAccumulatedGradient) {
    mt::Tensor x({2}, true);
    mt::Tensor grad1({2});
    mt::Tensor grad2({2});
    set_values(grad1, {3.0f, 5.0f});
    set_values(grad2, {2.0f, 4.0f});

    x.accumulate_grad(grad1);
    x.accumulate_grad(grad2);
    x.zero_grad();

    x.accumulate_grad(grad1);
    expect_values(x.grad(), {3.0f, 5.0f});
}

TEST(AutoGradStorageTest, AccumulatedGradientDoesNotAliasIncomingTensor) {
    mt::Tensor x({2}, true);
    mt::Tensor incoming({2});
    set_values(incoming, {3.0f, 5.0f});

    x.accumulate_grad(incoming);
    set_values(incoming, {100.0f, 200.0f});

    expect_values(x.grad(), {3.0f, 5.0f});
}

TEST(AutoGradStorageTest, StoredGradientDoesNotBuildGraph) {
    mt::Tensor x({2}, true);
    mt::Tensor incoming({2});
    set_values(incoming, {3.0f, 5.0f});

    x.accumulate_grad(incoming);

    mt::Tensor grad = x.grad();
    EXPECT_FALSE(grad.requires_grad());
    EXPECT_TRUE(grad.is_leaf());
    EXPECT_EQ(grad.grad_fn(), nullptr);
}

TEST(AutoGradStorageTest, AccumulateGradientRejectsMismatchedShape) {
    mt::Tensor x({2}, true);
    mt::Tensor incoming({1});

    EXPECT_THROW(x.accumulate_grad(incoming), std::runtime_error);
}

// -----------------------------------------------------------------------------
// backward() API
// -----------------------------------------------------------------------------

TEST(BackwardApiTest, RequiresGradientTracking) {
    mt::Tensor x({1}, false);

    EXPECT_THROW(x.backward(), std::runtime_error);
}

TEST(BackwardApiTest, RequiresSingleElementOutput) {
    mt::Tensor x({2}, true);

    EXPECT_THROW(x.backward(), std::runtime_error);
}

TEST(BackwardApiTest, ScalarLikeOutputSeedsGradientWithOne) {
    mt::Tensor x({1}, true);
    x.storage_at(0) = 4.0f;

    mt::Tensor y = x + x;
    y.backward();

    ASSERT_TRUE(x.has_grad());
    EXPECT_FLOAT_EQ(x.grad().item(), 2.0f);
}

// -----------------------------------------------------------------------------
// reduce_to_shape
// -----------------------------------------------------------------------------

TEST(ReduceToShapeTest, SameShapeLeavesValuesUnchanged) {
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    mt::Tensor out = mt::reduce_to_shape(grad, {2, 3});

    expect_shape(out, {2, 3});
    expect_values(out, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

TEST(ReduceToShapeTest, ReducesLastDimension) {
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    mt::Tensor out = mt::reduce_to_shape(grad, {2, 1});

    expect_shape(out, {2, 1});
    expect_values(out, {6.0f, 15.0f});
}

TEST(ReduceToShapeTest, ReducesBroadcastedFirstDimension) {
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    mt::Tensor out = mt::reduce_to_shape(grad, {1, 3});

    expect_shape(out, {1, 3});
    expect_values(out, {5.0f, 7.0f, 9.0f});
}

TEST(ReduceToShapeTest, ReducesLeadingDimension) {
    mt::Tensor grad = mt::arange(1.0f, 25.0f).reshape({2, 3, 4});

    mt::Tensor out = mt::reduce_to_shape(grad, {3, 4});

    expect_shape(out, {3, 4});
    EXPECT_FLOAT_EQ(out.at({0, 0}), 14.0f);
    EXPECT_FLOAT_EQ(out.at({2, 3}), 36.0f);
}

TEST(ReduceToShapeTest, RejectsIncompatibleShape) {
    mt::Tensor grad = mt::ones({2, 3});

    EXPECT_THROW(mt::reduce_to_shape(grad, {2, 4}), std::runtime_error);
}

// -----------------------------------------------------------------------------
// AddBackwardNode
// -----------------------------------------------------------------------------

TEST(AddBackwardTest, PropagatesGradientToBothInputs) {
    mt::Tensor x({1}, true);
    mt::Tensor y({1}, true);
    x.storage_at(0) = 2.0f;
    y.storage_at(0) = 3.0f;

    mt::Tensor out = x + y;
    out.backward();

    EXPECT_FLOAT_EQ(x.grad().item(), 1.0f);
    EXPECT_FLOAT_EQ(y.grad().item(), 1.0f);
}

TEST(AddBackwardTest, PropagatesOnlyToInputsRequiringGrad) {
    mt::Tensor x({1}, true);
    mt::Tensor y({1}, false);

    mt::Tensor out = x + y;
    out.backward();

    EXPECT_FLOAT_EQ(x.grad().item(), 1.0f);
    EXPECT_FALSE(y.has_grad());
}

TEST(AddBackwardTest, ReducesGradientForBroadcastInputs) {
    mt::Tensor a = mt::ones({2, 1}, true);
    mt::Tensor b = mt::ones({1, 3}, true);
    mt::Tensor out = a + b;
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    out.backward_propagate(grad);

    expect_shape(a.grad(), {2, 1});
    expect_values(a.grad(), {6.0f, 15.0f});
    expect_shape(b.grad(), {1, 3});
    expect_values(b.grad(), {5.0f, 7.0f, 9.0f});
}

TEST(AddBackwardTest, PropagatesCustomGradientForSameShapeInputs) {
    mt::Tensor a = mt::ones({2, 2}, true);
    mt::Tensor b = mt::ones({2, 2}, true);
    mt::Tensor grad = mt::arange(1.0f, 5.0f).reshape({2, 2});

    mt::Tensor out = a + b;
    out.backward_propagate(grad);

    expect_values(a.grad(), {1.0f, 2.0f, 3.0f, 4.0f});
    expect_values(b.grad(), {1.0f, 2.0f, 3.0f, 4.0f});
}

TEST(AddBackwardTest, DoesNotPropagateToBroadcastInputWithoutGrad) {
    mt::Tensor a = mt::ones({2, 1}, true);
    mt::Tensor b = mt::ones({1, 3}, false);
    mt::Tensor out = a + b;
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    out.backward_propagate(grad);

    expect_values(a.grad(), {6.0f, 15.0f});
    EXPECT_FALSE(b.has_grad());
}

TEST(AddBackwardTest, AccumulatesGradientFromSeparatePaths) {
    mt::Tensor x({1}, true);
    mt::Tensor y({1}, true);

    mt::Tensor left = x + y;
    mt::Tensor right = x + y;
    mt::Tensor loss = left + right;
    loss.backward();

    EXPECT_FLOAT_EQ(x.grad().item(), 2.0f);
    EXPECT_FLOAT_EQ(y.grad().item(), 2.0f);
}

TEST(AddBackwardTest, PropagatesThroughIntermediateTensorWithBroadcast) {
    mt::Tensor x = mt::ones({2, 1}, true);
    mt::Tensor y = mt::ones({1, 3}, true);

    mt::Tensor z = x + y;
    mt::Tensor out = z + y;
    out.backward_propagate(mt::ones({2, 3}));

    expect_values(x.grad(), {3.0f, 3.0f});
    expect_values(y.grad(), {4.0f, 4.0f, 4.0f});
}

TEST(AddBackwardTest, AccumulatesWhenIntermediateIsUsedTwice) {
    mt::Tensor x = mt::ones({2, 1}, true);
    mt::Tensor y = mt::ones({1, 3}, true);

    mt::Tensor z = x + y;
    mt::Tensor out = z + z;
    out.backward_propagate(mt::ones({2, 3}));

    expect_values(x.grad(), {6.0f, 6.0f});
    expect_values(y.grad(), {4.0f, 4.0f, 4.0f});
}

// -----------------------------------------------------------------------------
// SubBackwardNode
// -----------------------------------------------------------------------------

TEST(SubBackwardTest, PropagatesGradientWithCorrectSigns) {
    mt::Tensor a({3}, true);
    mt::Tensor b({3}, true);
    mt::Tensor grad({3});
    set_values(grad, {2.0f, -3.0f, 4.0f});

    mt::Tensor out = a - b;
    out.backward_propagate(grad);

    expect_values(a.grad(), {2.0f, -3.0f, 4.0f});
    expect_values(b.grad(), {-2.0f, 3.0f, -4.0f});
}

TEST(SubBackwardTest, ReducesGradientForBroadcastInputs) {
    mt::Tensor a = mt::ones({2, 1}, true);
    mt::Tensor b = mt::ones({1, 3}, true);
    mt::Tensor out = a - b;
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    out.backward_propagate(grad);

    expect_values(a.grad(), {6.0f, 15.0f});
    expect_values(b.grad(), {-5.0f, -7.0f, -9.0f});
}

// -----------------------------------------------------------------------------
// MulBackwardNode
// -----------------------------------------------------------------------------

TEST(MulBackwardTest, PropagatesGradientToBothInputs) {
    mt::Tensor a({3}, true);
    mt::Tensor b({3}, true);
    set_values(a, {2.0f, 3.0f, 4.0f});
    set_values(b, {5.0f, 6.0f, 7.0f});
    mt::Tensor grad({3});
    set_values(grad, {1.0f, 2.0f, 3.0f});

    mt::Tensor out = a * b;
    out.backward_propagate(grad);

    expect_values(a.grad(), {5.0f, 12.0f, 21.0f});
    expect_values(b.grad(), {2.0f, 6.0f, 12.0f});
}

TEST(MulBackwardTest, ReducesGradientAfterElementwiseMultiplication) {
    mt::Tensor a({2, 1}, true);
    mt::Tensor b({1, 3}, true);
    set_values(a, {2.0f, 3.0f});
    set_values(b, {4.0f, 5.0f, 6.0f});
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    mt::Tensor out = a * b;
    out.backward_propagate(grad);

    expect_shape(a.grad(), {2, 1});
    expect_values(a.grad(), {32.0f, 77.0f});
    expect_shape(b.grad(), {1, 3});
    expect_values(b.grad(), {14.0f, 19.0f, 24.0f});
}

TEST(MulBackwardTest, DoesNotPropagateToInputWithoutGrad) {
    mt::Tensor a({2}, true);
    mt::Tensor b({2}, false);
    set_values(a, {2.0f, 3.0f});
    set_values(b, {4.0f, 5.0f});

    mt::Tensor out = a * b;
    mt::Tensor grad = mt::arange(1.0f, 3.0f).reshape({2});
    out.backward_propagate(grad);

    expect_values(a.grad(), {4.0f, 10.0f});
    EXPECT_FALSE(b.has_grad());
}

TEST(MulBackwardTest, GradientComputationDoesNotBuildGraph) {
    mt::Tensor a({2}, true);
    mt::Tensor b({2}, true);
    set_values(a, {2.0f, 3.0f});
    set_values(b, {4.0f, 5.0f});

    mt::Tensor out = a * b;
    mt::Tensor grad = mt::arange(1.0f, 3.0f).reshape({2});
    out.backward_propagate(grad);

    mt::Tensor grad_a = a.grad();
    mt::Tensor grad_b = b.grad();
    EXPECT_FALSE(grad_a.requires_grad());
    EXPECT_FALSE(grad_b.requires_grad());
    EXPECT_EQ(grad_a.grad_fn(), nullptr);
    EXPECT_EQ(grad_b.grad_fn(), nullptr);
    EXPECT_TRUE(mt::is_grad_enabled());
}

// -----------------------------------------------------------------------------
// DivBackwardNode
// -----------------------------------------------------------------------------

TEST(DivBackwardTest, PropagatesGradientToBothInputs) {
    mt::Tensor a({3}, true);
    mt::Tensor b({3}, true);
    set_values(a, {8.0f, 12.0f, 20.0f});
    set_values(b, {2.0f, 3.0f, 4.0f});
    mt::Tensor grad({3});
    set_values(grad, {1.0f, 2.0f, 3.0f});

    mt::Tensor out = a / b;
    out.backward_propagate(grad);

    expect_values(a.grad(), {0.5f, 2.0f / 3.0f, 0.75f});
    expect_values(b.grad(), {-2.0f, -8.0f / 3.0f, -3.75f});
}

TEST(DivBackwardTest, ReducesGradientAfterElementwiseDivision) {
    mt::Tensor a({2, 1}, true);
    mt::Tensor b({1, 3}, true);
    set_values(a, {8.0f, 12.0f});
    set_values(b, {2.0f, 3.0f, 4.0f});
    mt::Tensor grad = mt::arange(1.0f, 7.0f).reshape({2, 3});

    mt::Tensor out = a / b;
    out.backward_propagate(grad);

    expect_values(a.grad(), {0.5f + 2.0f / 3.0f + 0.75f, 2.0f + 5.0f / 3.0f + 1.5f});
    expect_values(b.grad(), {-14.0f, -76.0f / 9.0f, -6.0f});
}

// -----------------------------------------------------------------------------
// NoGradGuard / gradient-recording state
// -----------------------------------------------------------------------------

TEST(NoGradGuardTest, GradientRecordingIsEnabledByDefault) { EXPECT_TRUE(mt::is_grad_enabled()); }

TEST(NoGradGuardTest, DisablesGradientRecordingWithinScope) {
    {
        mt::NoGradGuard guard;
        EXPECT_FALSE(mt::is_grad_enabled());
    }
    EXPECT_TRUE(mt::is_grad_enabled());
}

TEST(NoGradGuardTest, RestoresPreviouslyDisabledState) {
    mt::set_grad_enabled(false);
    {
        mt::NoGradGuard guard;
        EXPECT_FALSE(mt::is_grad_enabled());
    }

    EXPECT_FALSE(mt::is_grad_enabled());
    mt::set_grad_enabled(true);  // Avoid leaking state into other tests.
}

TEST(NoGradGuardTest, NestedGuardsRestoreStateCorrectly) {
    {
        mt::NoGradGuard outer;
        EXPECT_FALSE(mt::is_grad_enabled());
        {
            mt::NoGradGuard inner;
            EXPECT_FALSE(mt::is_grad_enabled());
        }
        EXPECT_FALSE(mt::is_grad_enabled());
    }
    EXPECT_TRUE(mt::is_grad_enabled());
}

TEST(NoGradGuardTest, OperationsDoNotBuildGraphWithinScope) {
    mt::Tensor a({2}, true);
    mt::Tensor b({2}, true);

    {
        mt::NoGradGuard guard;
        mt::Tensor out = a * b;
        EXPECT_FALSE(out.requires_grad());
        EXPECT_TRUE(out.is_leaf());
        EXPECT_EQ(out.grad_fn(), nullptr);
    }

    mt::Tensor out = a * b;
    EXPECT_TRUE(out.requires_grad());
    EXPECT_FALSE(out.is_leaf());
    ASSERT_NE(out.grad_fn(), nullptr);
}

// -----------------------------------------------------------------------------
// ReluBackwardNode
// -----------------------------------------------------------------------------

TEST(ReluBackwardTest, PropagatesGradientAccordingToInputSign) {
    mt::Tensor x({4}, true);
    set_values(x, {-2.0f, 0.0f, 3.0f, 5.0f});

    mt::Tensor out = x.relu();
    out.backward_propagate(mt::ones({4}));

    expect_values(x.grad(), {0.0f, 0.0f, 1.0f, 1.0f});
}

TEST(ReluBackwardTest, PropagatesThroughIntermediateUsedTwice) {
    mt::Tensor x({3}, true);
    set_values(x, {-1.0f, 2.0f, 4.0f});

    mt::Tensor z = x.relu();
    mt::Tensor out = z + z;
    out.backward_propagate(mt::ones({3}));

    expect_values(x.grad(), {0.0f, 2.0f, 2.0f});
}

TEST(ReluBackwardTest, DoesNotBuildGraphWhenInputDoesNotRequireGrad) {
    mt::Tensor x = mt::ones({3}, false);
    mt::Tensor out = x.relu();

    EXPECT_FALSE(out.requires_grad());
    EXPECT_TRUE(out.is_leaf());
    EXPECT_EQ(out.grad_fn(), nullptr);
}