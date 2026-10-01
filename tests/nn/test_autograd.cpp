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