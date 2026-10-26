#include "mini_torch/autograd_nodes.h"

#include <stdexcept>
#include <vector>

#include "mini_torch/autograd.h"
#include "mini_torch/tensor.h"
#include "mini_torch/tensor_iterator.h"

namespace mt {
Tensor reduce_to_shape(const Tensor& grad, const std::vector<size_t>& target_shape) {
    std::vector<size_t> grad_shape = grad.shape();
    size_t grad_ndim = grad_shape.size();
    size_t target_ndim = target_shape.size();

    if (grad_ndim < target_ndim) {
        throw std::runtime_error(
            "Gradient tensor must have at least as many dimensions "
            "as the target shape");
    }

    size_t d = grad_ndim - target_ndim;

    // Validate broadcasting compatibility.
    for (size_t dim = 0; dim < target_ndim; ++dim) {
        const size_t grad_dim = grad_shape[d + dim];
        const size_t target_dim = target_shape[dim];

        if (target_dim != 1 && target_dim != grad_dim) {
            throw std::runtime_error("Gradient shape is incompatible with target shape");
        }
    }

    Tensor out(target_shape, false);

    IndexIterator iter(grad_shape);
    while (!iter.done()) {
        std::vector<size_t> grad_index = iter.index();
        std::vector<size_t> out_index(target_ndim);

        for (size_t dim = 0; dim < target_ndim; ++dim) {
            out_index[dim] = target_shape[dim] > 1 ? grad_index[d + dim] : 0;
        }

        out.at(out_index) += grad.at(grad_index);

        iter.next();
    }

    return out;
}

AddBackwardNode::AddBackwardNode(const mt::Tensor& lhs, const mt::Tensor& rhs)
    : lhs_(lhs), rhs_(rhs) {};

const Tensor& AddBackwardNode::lhs() const { return lhs_; }

const Tensor& AddBackwardNode::rhs() const { return rhs_; }

std::string AddBackwardNode::name() const { return "AddBackward"; }

void AddBackwardNode::backward(const Tensor& grad_output) {
    if (lhs_.requires_grad()) {
        Tensor grad_lhs = reduce_to_shape(grad_output, lhs_.shape());
        lhs_.backward_propagate(grad_lhs);
    }

    if (rhs_.requires_grad()) {
        Tensor grad_rhs = reduce_to_shape(grad_output, rhs_.shape());
        rhs_.backward_propagate(grad_rhs);
    }
}

SubBackwardNode::SubBackwardNode(const mt::Tensor& lhs, const mt::Tensor& rhs)
    : lhs_(lhs), rhs_(rhs) {};

const Tensor& SubBackwardNode::lhs() const { return lhs_; }

const Tensor& SubBackwardNode::rhs() const { return rhs_; }

std::string SubBackwardNode::name() const { return "SubBackward"; }

void SubBackwardNode::backward(const Tensor& grad_output) {
    if (lhs_.requires_grad()) {
        Tensor grad_lhs = reduce_to_shape(grad_output, lhs_.shape());
        lhs_.backward_propagate(grad_lhs);
    }

    if (rhs_.requires_grad()) {
        Tensor grad_rhs = -1.0f * reduce_to_shape(grad_output, rhs_.shape());
        rhs_.backward_propagate(grad_rhs);
    }
}

MulBackwardNode::MulBackwardNode(const mt::Tensor& lhs, const mt::Tensor& rhs)
    : lhs_(lhs), rhs_(rhs) {};

const Tensor& MulBackwardNode::lhs() const { return lhs_; }

const Tensor& MulBackwardNode::rhs() const { return rhs_; }

std::string MulBackwardNode::name() const { return "MulBackward"; }

void MulBackwardNode::backward(const Tensor& grad_output) {
    if (lhs_.requires_grad()) {
        Tensor grad_lhs = reduce_to_shape(grad_output * rhs_, lhs_.shape());
        lhs_.backward_propagate(grad_lhs);
    }

    if (rhs_.requires_grad()) {
        Tensor grad_rhs = reduce_to_shape(grad_output * lhs_, rhs_.shape());
        rhs_.backward_propagate(grad_rhs);
    }
}

DivBackwardNode::DivBackwardNode(const mt::Tensor& lhs, const mt::Tensor& rhs)
    : lhs_(lhs), rhs_(rhs) {};

const Tensor& DivBackwardNode::lhs() const { return lhs_; }

const Tensor& DivBackwardNode::rhs() const { return rhs_; }

std::string DivBackwardNode::name() const { return "DivBackward"; }

void DivBackwardNode::backward(const Tensor& grad_output) {
    if (lhs_.requires_grad()) {
        Tensor grad_lhs = reduce_to_shape(grad_output / rhs_, lhs_.shape());
        lhs_.backward_propagate(grad_lhs);
    }

    if (rhs_.requires_grad()) {
        Tensor grad_rhs =
            -1.0f * reduce_to_shape(grad_output * (lhs_ / (rhs_ * rhs_)), rhs_.shape());
        rhs_.backward_propagate(grad_rhs);
    }
}

ReluBackwardNode::ReluBackwardNode(const Tensor& input) : input_(input) {};

const Tensor& ReluBackwardNode::input() const { return input_; }

std::string ReluBackwardNode::name() const { return "ReLUBackward"; }

void ReluBackwardNode::backward(const Tensor& grad_output) {
    if (input_.requires_grad()) {
        Tensor grad_input(input_.shape(), false);

        IndexIterator iter(input_.shape());
        while (!iter.done()) {
            std::vector<size_t> index = iter.index();
            grad_input.at(index) = input_.at(index) > 0.0f ? grad_output.at(index) : 0.0f;

            iter.next();
        }

        input_.backward_propagate(grad_input);
    }
}
}  // namespace mt