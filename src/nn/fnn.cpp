#include "pinn/nn/fnn.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

#include "pinn/nn/initialization.hpp"

namespace pinn::nn {

namespace {
std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}
}  // namespace

NetworkArchitecture architecture_from_string(const std::string& name) {
    const auto lower = to_lower(name);
    if (lower == "fnn" || lower == "mlp" || lower == "dense") {
        return NetworkArchitecture::kFnn;
    }
    if (lower == "resnet" || lower == "residual") {
        return NetworkArchitecture::kResNet;
    }
    if (lower == "cnn" || lower == "conv" || lower == "convolutional") {
        return NetworkArchitecture::kCnn;
    }
    if (lower == "transformer" || lower == "attention" || lower == "attn") {
        return NetworkArchitecture::kTransformer;
    }
    throw std::invalid_argument{"Unsupported network architecture: " + name};
}

std::string to_string(NetworkArchitecture arch) {
    switch (arch) {
        case NetworkArchitecture::kFnn:
            return "fnn";
        case NetworkArchitecture::kResNet:
            return "resnet";
        case NetworkArchitecture::kCnn:
            return "cnn";
        case NetworkArchitecture::kTransformer:
            return "transformer";
        default:
            return "unknown";
    }
}

FnnImpl::FnnImpl(std::vector<int> layer_sizes,
                 ActivationFn activation,
                 InitType init_type,
                 Scalar bias_init,
                 int64_t seed,
                 ArchitectureConfig arch_config)
    : layer_sizes_{std::move(layer_sizes)},
      activation_{std::move(activation)},
      init_type_{init_type},
      bias_init_{bias_init},
      input_dim_{0},
      output_dim_{0},
      arch_config_{std::move(arch_config)},
      architecture_{arch_config_.architecture},
      use_adaptive_activation_{arch_config_.use_adaptive_activation},
      adaptive_activation_init_{arch_config_.adaptive_activation_init},
      activation_slots_{0},
      resnet_block_count_{0},
      cnn_flatten_dim_{0},
      transformer_embed_dim_{arch_config_.transformer_embed_dim} {
    if (layer_sizes_.size() < 2) {
        throw std::invalid_argument{"Network requires at least input and output layers"};
    }
    input_dim_ = layer_sizes_.front();
    output_dim_ = layer_sizes_.back();

    if (seed >= 0) {
        torch::manual_seed(seed);
    }

    layers_ = register_module("layers", torch::nn::ModuleList());

    switch (architecture_) {
        case NetworkArchitecture::kFnn:
            build_plain_network();
            break;
        case NetworkArchitecture::kResNet:
            build_resnet_network();
            break;
        case NetworkArchitecture::kCnn:
            build_cnn_network();
            break;
        case NetworkArchitecture::kTransformer:
            build_transformer_network();
            break;
        default:
            TORCH_CHECK(false, "Unsupported network architecture");
    }

    torch::optional<Scalar> bias{bias_init_};
    apply_initialization(*this, init_type_, bias);
    initialize_adaptive_parameters();
}

Tensor FnnImpl::forward(const Tensor& x) {
    switch (architecture_) {
        case NetworkArchitecture::kFnn:
            return forward_plain(x);
        case NetworkArchitecture::kResNet:
            return forward_resnet(x);
        case NetworkArchitecture::kCnn:
            return forward_cnn(x);
        case NetworkArchitecture::kTransformer:
            return forward_transformer(x);
        default:
            TORCH_CHECK(false, "Unsupported network architecture");
    }
}

void FnnImpl::build_plain_network() {
    for (size_t i = 0; i + 1 < layer_sizes_.size(); ++i) {
        auto linear = torch::nn::Linear(torch::nn::LinearOptions(layer_sizes_[i], layer_sizes_[i + 1]));
        layers_->push_back(linear);
    }
    activation_slots_ = layers_->size() > 0 ? layers_->size() - 1 : 0;
}

void FnnImpl::build_resnet_network() {
    if (layer_sizes_.size() < 3) {
        throw std::invalid_argument{"ResNet architecture requires at least one hidden layer"};
    }
    const int hidden_dim = layer_sizes_[1];
    if (hidden_dim <= 0) {
        throw std::invalid_argument{"ResNet hidden dimension must be positive"};
    }

    resnet_input_ = register_module("resnet_input", torch::nn::Linear(torch::nn::LinearOptions(input_dim_, hidden_dim)));
    resnet_output_ = register_module("resnet_output", torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, output_dim_)));
    resnet_blocks_ = register_module("resnet_blocks", torch::nn::ModuleList());

    resnet_block_count_ = arch_config_.resnet_blocks > 0 ? arch_config_.resnet_blocks : static_cast<int>(layer_sizes_.size()) - 2;
    if (resnet_block_count_ <= 0) {
        resnet_block_count_ = 1;
    }

    for (int i = 0; i < resnet_block_count_; ++i) {
        auto fc1 = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim));
        auto fc2 = torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, hidden_dim));
        resnet_blocks_->push_back(fc1);
        resnet_blocks_->push_back(fc2);
    }

    activation_slots_ = 1 + static_cast<size_t>(resnet_block_count_) + (resnet_block_count_ > 1 ? resnet_block_count_ - 1 : 0);
}

void FnnImpl::build_cnn_network() {
    if (input_dim_ <= 0) {
        throw std::invalid_argument{"CNN architecture requires positive input dimension"};
    }

    const int cnn_layers = std::max(1, arch_config_.cnn_layers);
    const int channels = std::max(1, arch_config_.cnn_channels);
    int kernel = std::max(1, arch_config_.cnn_kernel_size);
    if (kernel > input_dim_) {
        kernel = input_dim_;
    }

    cnn_convs_ = register_module("cnn_convs", torch::nn::ModuleList());
    int in_channels = 1;
    for (int i = 0; i < cnn_layers; ++i) {
        auto conv = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, channels, kernel).padding(kernel / 2));
        cnn_convs_->push_back(conv);
        in_channels = channels;
    }

    cnn_flatten_dim_ = in_channels * input_dim_;

    if (layer_sizes_.size() < 2) {
        throw std::invalid_argument{"CNN head requires at least output layer"};
    }

    cnn_head_ = register_module("cnn_head", torch::nn::ModuleList());
    int prev_dim = cnn_flatten_dim_;
    for (size_t i = 1; i < layer_sizes_.size(); ++i) {
        auto linear = torch::nn::Linear(torch::nn::LinearOptions(prev_dim, layer_sizes_[i]));
        cnn_head_->push_back(linear);
        prev_dim = layer_sizes_[i];
    }

    const size_t head_activations = cnn_head_->size() > 0 ? cnn_head_->size() - 1 : 0;
    activation_slots_ = static_cast<size_t>(cnn_layers) + head_activations;
}

void FnnImpl::build_transformer_network() {
    if (input_dim_ <= 0) {
        throw std::invalid_argument{"Transformer architecture requires positive input dimension"};
    }

    transformer_embed_dim_ = arch_config_.transformer_embed_dim > 0 ? arch_config_.transformer_embed_dim
                                                                    : (layer_sizes_.size() > 1 ? layer_sizes_[1] : 64);
    if (transformer_embed_dim_ % std::max(1, arch_config_.transformer_heads) != 0) {
        throw std::invalid_argument{"Transformer embedding dimension must be divisible by number of heads"};
    }

    transformer_embedding_ = register_module("transformer_embedding",
                                             torch::nn::Linear(torch::nn::LinearOptions(1, transformer_embed_dim_)));

    auto encoder_layer = torch::nn::TransformerEncoderLayer(
        torch::nn::TransformerEncoderLayerOptions(transformer_embed_dim_, std::max(1, arch_config_.transformer_heads))
            .dim_feedforward(std::max(arch_config_.transformer_ffn_dim, transformer_embed_dim_)));
    const int encoder_layers = std::max(1, arch_config_.transformer_layers);
    transformer_encoder_ = register_module("transformer_encoder", torch::nn::TransformerEncoder(encoder_layer, encoder_layers));

    if (layer_sizes_.size() < 2) {
        throw std::invalid_argument{"Transformer head requires at least output layer"};
    }

    transformer_head_ = register_module("transformer_head", torch::nn::ModuleList());
    int prev_dim = transformer_embed_dim_;
    for (size_t i = 1; i < layer_sizes_.size(); ++i) {
        auto linear = torch::nn::Linear(torch::nn::LinearOptions(prev_dim, layer_sizes_[i]));
        transformer_head_->push_back(linear);
        prev_dim = layer_sizes_[i];
    }

    activation_slots_ = transformer_head_->size() > 0 ? transformer_head_->size() - 1 : 0;
}

void FnnImpl::initialize_adaptive_parameters() {
    adaptive_scales_.clear();
    if (!use_adaptive_activation_ || activation_slots_ == 0) {
        return;
    }

    for (size_t i = 0; i < activation_slots_; ++i) {
        auto name = "adaptive_scale_" + std::to_string(i);
        auto param = register_parameter(name, torch::full({1}, static_cast<float>(adaptive_activation_init_)));
        adaptive_scales_.push_back(param);
    }
}

Tensor FnnImpl::forward_plain(const Tensor& x) {
    TORCH_CHECK(layers_->size() >= 2, "FNN requires at least two layers");
    Tensor out = x;
    const size_t last_index = layers_->size() - 1;
    size_t activation_slot = 0;
    for (size_t i = 0; i < layers_->size(); ++i) {
        auto* linear = (*layers_)[i]->as<torch::nn::LinearImpl>();
        TORCH_CHECK(linear != nullptr, "Expected Linear layer");
        out = linear->forward(out);
        if (i != last_index) {
            out = apply_activation(out, activation_slot++);
        }
    }
    return out;
}

Tensor FnnImpl::forward_resnet(const Tensor& x) {
    TORCH_CHECK(resnet_input_ && resnet_output_ && resnet_blocks_, "ResNet modules are not initialized");
    Tensor out = resnet_input_->forward(x);
    size_t activation_slot = 0;
    out = apply_activation(out, activation_slot++);

    for (int block = 0; block < resnet_block_count_; ++block) {
        const size_t base = static_cast<size_t>(block) * 2;
        auto* fc1 = (*resnet_blocks_)[base]->as<torch::nn::LinearImpl>();
        auto* fc2 = (*resnet_blocks_)[base + 1]->as<torch::nn::LinearImpl>();
        TORCH_CHECK(fc1 && fc2, "ResNet block expects Linear layers");
        auto residual = out;
        auto z = fc1->forward(out);
        z = apply_activation(z, activation_slot++);
        z = fc2->forward(z);
        out = z + residual;
        if (block + 1 < resnet_block_count_) {
            out = apply_activation(out, activation_slot++);
        }
    }

    return resnet_output_->forward(out);
}

Tensor FnnImpl::forward_cnn(const Tensor& x) {
    TORCH_CHECK(cnn_convs_ && cnn_head_, "CNN modules are not initialized");
    TORCH_CHECK(x.dim() == 2, "CNN expects 2D input (batch, features)");

    Tensor out = x.unsqueeze(1);
    size_t activation_slot = 0;
    for (const auto& module : *cnn_convs_) {
        auto* conv = module->as<torch::nn::Conv1dImpl>();
        TORCH_CHECK(conv != nullptr, "CNN conv block expects Conv1d");
        out = conv->forward(out);
        out = apply_activation(out, activation_slot++);
    }

    out = out.flatten(1);
    for (size_t i = 0; i < cnn_head_->size(); ++i) {
        auto* linear = (*cnn_head_)[i]->as<torch::nn::LinearImpl>();
        TORCH_CHECK(linear != nullptr, "CNN head expects Linear layers");
        out = linear->forward(out);
        if (i + 1 != cnn_head_->size()) {
            out = apply_activation(out, activation_slot++);
        }
    }

    return out;
}

Tensor FnnImpl::forward_transformer(const Tensor& x) {
    TORCH_CHECK(transformer_embedding_ && transformer_encoder_ && transformer_head_,
                "Transformer modules are not initialized");
    TORCH_CHECK(x.dim() == 2, "Transformer expects 2D input (batch, features)");

    auto batch = x.size(0);
    auto n_features = x.size(1);
    auto tokens = x.unsqueeze(-1);
    tokens = tokens.view({batch * n_features, 1});
    tokens = transformer_embedding_->forward(tokens);
    tokens = tokens.view({batch, n_features, transformer_embed_dim_}).transpose(0, 1);
    auto encoded = transformer_encoder_->forward(tokens).transpose(0, 1);
    auto pooled = encoded.mean(1);

    Tensor out = pooled;
    size_t activation_slot = 0;
    for (size_t i = 0; i < transformer_head_->size(); ++i) {
        auto* linear = (*transformer_head_)[i]->as<torch::nn::LinearImpl>();
        TORCH_CHECK(linear != nullptr, "Transformer head expects Linear layers");
        out = linear->forward(out);
        if (i + 1 != transformer_head_->size()) {
            out = apply_activation(out, activation_slot++);
        }
    }

    return out;
}

Tensor FnnImpl::apply_activation(const Tensor& value, size_t slot) const {
    if (use_adaptive_activation_ && slot < adaptive_scales_.size()) {
        return activation_(adaptive_scales_[slot] * value);
    }
    return activation_(value);
}

}  // namespace pinn::nn
