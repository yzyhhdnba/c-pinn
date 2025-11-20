#pragma once

#include <string>
#include <vector>

#include "pinn/nn/activation.hpp"
#include "pinn/nn/initialization.hpp"
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

class FnnImpl : public torch::nn::Module {
  public:
    FnnImpl(std::vector<int> layer_sizes,
            ActivationFn activation,
            InitType init_type,
            Scalar bias_init,
          int64_t seed = -1,
          ArchitectureConfig arch_config = {});

    Tensor forward(const Tensor& x);

    int input_dim() const noexcept { return input_dim_; }
    int output_dim() const noexcept { return output_dim_; }

  private:
    void build_plain_network();
    void build_resnet_network();
    void build_cnn_network();
    void build_transformer_network();
    void initialize_adaptive_parameters();
    Tensor forward_plain(const Tensor& x);
    Tensor forward_resnet(const Tensor& x);
    Tensor forward_cnn(const Tensor& x);
    Tensor forward_transformer(const Tensor& x);
    Tensor apply_activation(const Tensor& value, size_t slot) const;

    std::vector<int> layer_sizes_;
    ActivationFn activation_;
    InitType init_type_;
    Scalar bias_init_;
    int input_dim_;
    int output_dim_;
    ArchitectureConfig arch_config_;
    NetworkArchitecture architecture_{NetworkArchitecture::kFnn};
    bool use_adaptive_activation_{false};
    Scalar adaptive_activation_init_{1.0};
    size_t activation_slots_{0};
    std::vector<Tensor> adaptive_scales_;
    int resnet_block_count_{0};
    int cnn_flatten_dim_{0};
    int transformer_embed_dim_{0};

    torch::nn::ModuleList layers_{nullptr};
    torch::nn::ModuleList resnet_blocks_{nullptr};
    torch::nn::ModuleList cnn_convs_{nullptr};
    torch::nn::ModuleList cnn_head_{nullptr};
    torch::nn::ModuleList transformer_head_{nullptr};
    torch::nn::Linear resnet_input_{nullptr};
    torch::nn::Linear resnet_output_{nullptr};
    torch::nn::Linear transformer_embedding_{nullptr};
    torch::nn::TransformerEncoder transformer_encoder_{nullptr};
};

TORCH_MODULE(Fnn);

}  // namespace pinn::nn
