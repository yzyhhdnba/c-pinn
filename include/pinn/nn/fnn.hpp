#pragma once

#include <vector>

#include "pinn/nn/activation.hpp"
#include "pinn/nn/initialization.hpp"
#include "pinn/types.hpp"

namespace pinn::nn {

class FnnImpl : public torch::nn::Module {
  public:
    FnnImpl(std::vector<int> layer_sizes,
            ActivationFn activation,
            InitType init_type,
            Scalar bias_init,
            int64_t seed = -1);

    Tensor forward(const Tensor& x);

    int input_dim() const noexcept { return input_dim_; }
    int output_dim() const noexcept { return output_dim_; }

  private:
    std::vector<int> layer_sizes_;
    ActivationFn activation_;
    InitType init_type_;
    Scalar bias_init_;
    int input_dim_;
    int output_dim_;
    torch::nn::ModuleList layers_;
};

TORCH_MODULE(Fnn);

}  // namespace pinn::nn
