#pragma once

#include <memory>
#include <string>

namespace mt {
class Tensor;

class Node {
   public:
    virtual ~Node() = default;

    virtual std::string name() const = 0;

    virtual void backward(const Tensor& grad_output) = 0;
};

}  // namespace mt