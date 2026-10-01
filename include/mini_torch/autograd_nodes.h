#pragma once

#include "mini_torch/autograd.h"
#include "mini_torch/tensor.h"

namespace mt {
class AddBackwardNode : public mt::Node {
   private:
    mt::Tensor lhs_;
    mt::Tensor rhs_;

   public:
    AddBackwardNode(const mt::Tensor& lhs, const mt::Tensor& rhs) : lhs_(lhs), rhs_(rhs) {};

    const Tensor& lhs() const { return lhs_; };

    const Tensor& rhs() const { return rhs_; };

    std::string name() const override { return "AddBackward"; }
};
}  // namespace mt