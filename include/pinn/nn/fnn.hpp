#pragma once

#include <string>
#include <vector>

#include "pinn/nn/activation.hpp"
#include "pinn/nn/initialization.hpp"
#include "pinn/nn/gemm.hpp"
#include "pinn/types.hpp"

namespace pinn::nn {

enum class NetworkArchitecture {
  kFnn,
  kResNet,
  kCnn,
  kTransformer
};

NetworkArchitecture architecture_from_string(const std::string& name);
std::string to_string(NetworkArchitecture arch);

struct ArchitectureConfig {
  NetworkArchitecture architecture{NetworkArchitecture::kFnn};
  int resnet_blocks{0};
  int cnn_channels{32};
  int cnn_layers{2};
  int cnn_kernel_size{3};
  int transformer_heads{4};
  int transformer_layers{2};
  int transformer_ffn_dim{128};
  int transformer_embed_dim{64};
  bool use_adaptive_activation{false};
  Scalar adaptive_activation_init{1.0};
};

class Linear {
  public:
    Linear() = default;
    Linear(int in_features,
           int out_features,
           InitType init_type,
           Scalar bias_init,
           pinn::core::Rng& rng);

    int in_features() const noexcept { return in_features_; }
    int out_features() const noexcept { return out_features_; }

    // x: [batch, in_features] (row-major). returns [batch, out_features]
    Tensor forward(const Tensor& x) const;

    // Backward pass
    // grad_output: [batch, out_features]
    // input: [batch, in_features] (cached from forward)
    // Returns grad_input: [batch, in_features]
    Tensor backward(const Tensor& grad_output, const Tensor& input);

    // Parameter accessors for flat mapping/training
    const Tensor& weight_t() const noexcept { return weight_t_; }
    const Tensor& bias() const noexcept { return bias_; }
    Tensor& weight_t() noexcept { return weight_t_; }
    Tensor& bias() noexcept { return bias_; }
    
    // Gradient accessors
    Tensor& grad_weight() noexcept { return grad_weight_; }
    Tensor& grad_bias() noexcept { return grad_bias_; }

    TYPE_VAL* weight_data() noexcept { return reinterpret_cast<TYPE_VAL*>(weight_t_.data_ptr<double>()); }
    TYPE_VAL* bias_data() noexcept { return reinterpret_cast<TYPE_VAL*>(bias_.data_ptr<double>()); }
    const TYPE_VAL* weight_data() const noexcept { return reinterpret_cast<const TYPE_VAL*>(weight_t_.data_ptr<double>()); }
    const TYPE_VAL* bias_data() const noexcept { return reinterpret_cast<const TYPE_VAL*>(bias_.data_ptr<double>()); }

  private:
    int in_features_{0};
    int out_features_{0};
    Tensor weight_t_;  // [in_features, out_features]
    Tensor bias_;      // [out_features]
    Tensor grad_weight_;
    Tensor grad_bias_;
};

class Fnn {
  public:
    Fnn(std::vector<int> layer_sizes,
        ActivationFn activation,
        ActivationFn activation_deriv,
        InitType init_type,
        Scalar bias_init,
        int64_t seed = -1,
        ArchitectureConfig arch_config = {});

    // Convenience constructor using string name for activation
    Fnn(std::vector<int> layer_sizes,
        const std::string& activation_name,
        InitType init_type,
        Scalar bias_init,
        int64_t seed = -1,
        ArchitectureConfig arch_config = {});

    // Forward pass (caches inputs for backward)
    Tensor forward(const Tensor& x);

    // Backward pass
    // grad_output: [batch, output_dim]
    // Returns grad_input: [batch, input_dim]
    Tensor backward(const Tensor& grad_output);

    int input_dim() const noexcept { return input_dim_; }
    int output_dim() const noexcept { return output_dim_; }

    int num_layers() const noexcept { return static_cast<int>(layers_.size()); }
    const std::vector<int>& layer_sizes() const noexcept { return layer_sizes_; }
    Linear& layer(int i) { return layers_.at(static_cast<size_t>(i)); }
    const Linear& layer(int i) const { return layers_.at(static_cast<size_t>(i)); }

  private:
    std::vector<int> layer_sizes_;
    ActivationFn activation_;
    ActivationFn activation_deriv_;
    InitType init_type_;
    Scalar bias_init_;
    int input_dim_{0};
    int output_dim_{0};
    ArchitectureConfig arch_config_;
    NetworkArchitecture architecture_{NetworkArchitecture::kFnn};
    std::vector<Linear> layers_;
    
    // Cache for backward pass
    std::vector<Tensor> inputs_;
    std::vector<Tensor> pre_activations_; // Not strictly needed for Tanh if we use output, but good for general
};

}  // namespace pinn::nn
