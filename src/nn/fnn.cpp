#include "pinn/nn/fnn.hpp"

#include <stdexcept>

#include <torch/torch.h>

#include "pinn/nn/initialization.hpp"

namespace pinn::nn {

FnnImpl::FnnImpl(std::vector<int> layer_sizes,
                 ActivationFn activation,
                 InitType init_type,
                 Scalar bias_init,
                 int64_t seed)
    : layer_sizes_{std::move(layer_sizes)},
      activation_{std::move(activation)},
      init_type_{init_type},
      bias_init_{bias_init},
      input_dim_{0},
      output_dim_{0} {
    if (layer_sizes_.size() < 2) {
        throw std::invalid_argument{"Feedforward network requires at least input and output layers"};
    }
    input_dim_ = layer_sizes_.front();
    output_dim_ = layer_sizes_.back();

    if (seed >= 0) {
        torch::manual_seed(seed);
    }

    register_module("layers", layers_);
    for (size_t i = 0; i + 1 < layer_sizes_.size(); ++i) {
        const auto in_features = layer_sizes_[i];
        const auto out_features = layer_sizes_[i + 1];
        auto linear = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features));
        layers_->push_back(linear);
    }

    torch::optional<Scalar> bias{bias_init_};
    apply_initialization(*this, init_type_, bias);
}

Tensor FnnImpl::forward(const Tensor& x) {
    Tensor out = x;
    const size_t last_index = layers_->size() - 1;
    for (size_t i = 0; i < layers_->size(); ++i) {
        auto module = (*layers_)[i];
        auto* linear = module->as<torch::nn::LinearImpl>();
        TORCH_CHECK(linear != nullptr, "ModuleList expected Linear layers only");
        out = linear->forward(out);
        if (i != last_index) {
            out = activation_(out);
        }
    }
    return out;
}

}  // namespace pinn::nn
