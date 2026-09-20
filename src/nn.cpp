#include "mini_torch/nn.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mini_torch/factory.h"
#include "mini_torch/tensor.h"

namespace mt {
namespace nn {
Parameter::Parameter() = default;

explicit Parameter::Parameter(Tensor data) : data_(std::move(data)) {}

Tensor& Parameter::data() { return data_; }

const Tensor& Parameter::data() const { return data_; }

Tensor ReLU::forward(const Tensor& x) { return x.relu(); }

Linear::Linear(size_t in_features, size_t out_features)
    : in_features_(in_features), out_features_(out_features) {
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Linear features must be greater than 0");
    }
    float bound = 1.0f / std::sqrt(in_features_);

    weight_ = Parameter(uniform({out_features_, in_features_}, -bound, bound));
    bias_ = Parameter(uniform({out_features_}, -bound, bound));
}

Tensor Linear::forward(const Tensor& x) {
    return x.matmul(weight_.data().transpose(0, 1)) + bias_.data();
}

std::vector<Parameter*> Linear::parameters() { return {&weight_, &bias_}; }

explicit Sequential::Sequential(std::vector<std::shared_ptr<Module>> layers)
    : layers_(std::move(layers)) {}

Tensor Sequential::forward(const Tensor& x) {
    Tensor out = x;
    for (auto layer : layers_) {
        out = layer->forward(out);
    }
    return out;
}

std::vector<Parameter*> Sequential::parameters() {
    std::vector<Parameter*> params;

    for (auto layer : layers_) {
        std::vector<Parameter*> layer_params = layer->parameters();

        for (auto param : layer_params) {
            params.push_back(param);
        }
    }

    return params;
}
}  // namespace nn
}  // namespace mt
