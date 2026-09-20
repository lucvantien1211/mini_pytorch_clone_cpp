#pragma once

#include <memory>
#include <vector>

#include "mini_torch/tensor.h"

namespace mt {
namespace nn {
class Parameter {
   private:
    Tensor data_;

   public:
    Parameter();
    explicit Parameter(Tensor data);

    Tensor& data();
    const Tensor& data() const;
};

class Module {
   public:
    virtual Tensor forward(const Tensor& x) = 0;

    virtual std::vector<Parameter*> parameters() { return {}; }

    virtual ~Module() = default;
};

class ReLU : public Module {
   public:
    Tensor forward(const Tensor& x) override;
};

class Linear : public Module {
   private:
    size_t in_features_;
    size_t out_features_;

    Parameter weight_;
    Parameter bias_;

   public:
    Linear(size_t in_features, size_t out_features);

    Tensor forward(const Tensor& x) override;

    std::vector<Parameter*> parameters() override;
};

class Sequential : public Module {
   private:
    std::vector<std::shared_ptr<Module>> layers_;

   public:
    explicit Sequential(std::vector<std::shared_ptr<Module>> layers);

    Tensor forward(const Tensor& x) override;

    std::vector<Parameter*> parameters() override;
};
}  // namespace nn
}  // namespace mt
