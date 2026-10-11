#include "mini_torch/autograd_nodes.h"

#include "mini_torch/autograd.h"
#include "mini_torch/tensor.h"

namespace mt {
AddBackwardNode::AddBackwardNode(const mt::Tensor& lhs, const mt::Tensor& rhs)
    : lhs_(lhs), rhs_(rhs) {};

const Tensor& AddBackwardNode::lhs() const { return lhs_; }

const Tensor& AddBackwardNode::rhs() const { return rhs_; }

std::string AddBackwardNode::name() const { return "AddBackward"; }

void AddBackwardNode::backward(const Tensor& grad_output) {
    if (lhs_.requires_grad()) {
        lhs_.backward_propagate(grad_output);
    }

    if (rhs_.requires_grad()) {
        rhs_.backward_propagate(grad_output);
    }
}
}  // namespace mt