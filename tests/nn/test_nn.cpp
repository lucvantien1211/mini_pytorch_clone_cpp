#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/nn.h"
#include "mini_torch/tensor.h"

// Parameter Tests
TEST(ParameterTest, ConstructFromTensor) {
    mt::Tensor t = mt::ones({2, 3});

    mt::nn::Parameter p(t);

    EXPECT_EQ(p.data().shape(), std::vector<size_t>({2, 3}));
}

TEST(ParameterTest, DataReturnsReference) {
    mt::nn::Parameter p(mt::zeros({2, 2}));

    p.data().storage_at(0) = 5.0f;

    EXPECT_FLOAT_EQ(p.data().storage_at(0), 5.0f);
}

// Linear Tests
TEST(LinearTest, ConstructorShapes) {
    mt::nn::Linear layer(4, 3);

    auto params = layer.parameters();

    ASSERT_EQ(params.size(), 2);

    EXPECT_EQ(params[0]->data().shape(), std::vector<size_t>({3, 4}));

    EXPECT_EQ(params[1]->data().shape(), std::vector<size_t>({3}));
}

TEST(LinearTest, ForwardOutputShape) {
    mt::nn::Linear layer(4, 3);

    mt::Tensor x = mt::ones({2, 4});

    mt::Tensor y = layer.forward(x);

    EXPECT_EQ(y.shape(), std::vector<size_t>({2, 3}));
}

TEST(LinearTest, ParametersCount) {
    mt::nn::Linear layer(10, 5);

    auto params = layer.parameters();

    EXPECT_EQ(params.size(), 2);
}

TEST(LinearTest, ForwardKnownValues) {
    mt::nn::Linear layer(2, 2);

    auto params = layer.parameters();

    auto& weight = params[0]->data();

    auto& bias = params[1]->data();

    // weight = [[1 2], [3 4]]
    weight.storage_at(0) = 1;
    weight.storage_at(1) = 2;
    weight.storage_at(2) = 3;
    weight.storage_at(3) = 4;

    // bias = [10 20]
    bias.storage_at(0) = 10;
    bias.storage_at(1) = 20;

    // x = [[1 1]
    mt::Tensor x({1, 2});

    x.storage_at(0) = 1;
    x.storage_at(1) = 1;

    mt::Tensor y = layer.forward(x);

    EXPECT_FLOAT_EQ(y.storage_at(0), 13.0f);

    EXPECT_FLOAT_EQ(y.storage_at(1), 27.0f);
}

TEST(ReLUTest, Forward) {
    mt::nn::ReLU relu;

    mt::Tensor x({3});

    x.storage_at(0) = -2;
    x.storage_at(1) = 0;
    x.storage_at(2) = 5;

    mt::Tensor y = relu.forward(x);

    EXPECT_FLOAT_EQ(y.storage_at(0), 0);
    EXPECT_FLOAT_EQ(y.storage_at(1), 0);
    EXPECT_FLOAT_EQ(y.storage_at(2), 5);
}

TEST(ReLUTest, NoParameters) {
    mt::nn::ReLU relu;

    EXPECT_TRUE(relu.parameters().empty());
}

TEST(SequentialTest, ForwardShape) {
    mt::nn::Sequential model({std::make_shared<mt::nn::Linear>(4, 8),
                              std::make_shared<mt::nn::ReLU>(),
                              std::make_shared<mt::nn::Linear>(8, 2)});

    mt::Tensor x = mt::ones({5, 4});

    mt::Tensor y = model.forward(x);

    EXPECT_EQ(y.shape(), std::vector<size_t>({5, 2}));
}

TEST(SequentialTest, ParametersCollected) {
    mt::nn::Sequential model({std::make_shared<mt::nn::Linear>(4, 8),
                              std::make_shared<mt::nn::ReLU>(),
                              std::make_shared<mt::nn::Linear>(8, 2)});

    auto params = model.parameters();

    EXPECT_EQ(params.size(), 4);
}

TEST(SequentialTest, NestedSequentialParameters) {
    auto block = std::make_shared<mt::nn::Sequential>(std::vector<std::shared_ptr<mt::nn::Module>>{
        std::make_shared<mt::nn::Linear>(4, 8), std::make_shared<mt::nn::ReLU>()});

    mt::nn::Sequential model({block, std::make_shared<mt::nn::Linear>(8, 2)});

    auto params = model.parameters();

    EXPECT_EQ(params.size(), 4);
}