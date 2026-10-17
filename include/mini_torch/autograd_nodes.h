#pragma once
#include <vector>

#include "mini_torch/autograd.h"
#include "mini_torch/tensor.h"

namespace mt {
Tensor reduce_to_shape(const Tensor& grad, const std::vector<size_t>& target_shape);

class AddBackwardNode : public mt::Node {
   private:
    mt::Tensor lhs_;
    mt::Tensor rhs_;

   public:
    AddBackwardNode(const mt::Tensor& lhs, const mt::Tensor& rhs);

    const Tensor& lhs() const;

    const Tensor& rhs() const;

    std::string name() const override;

    void backward(const Tensor& grad_output) override;
};
}  // namespace mt