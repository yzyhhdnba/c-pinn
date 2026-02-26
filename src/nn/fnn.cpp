
// Pure C backend neural network implementation.
//
// This file intentionally contains no LibTorch/torch::nn code.

#include "pinn/nn/fnn.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

#include "pinn/core/rng.hpp"

namespace pinn::nn {

namespace {
std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

void ensure_2d(const Tensor& x) {
    if (!x.defined()) {
        throw std::runtime_error("Fnn::forward received undefined tensor");
    }
    if (x.dim() != 2) {
        throw std::runtime_error("Fnn::forward expects 2D tensor (batch, features)");
    }
    if (x.dtype() != pinn::core::DType::kFloat64) {
        throw std::runtime_error("Fnn::forward currently supports float64 tensors only");
    }
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

Linear::Linear(int in_features, int out_features, InitType init_type, Scalar bias_init, pinn::core::Rng& rng)
    : in_features_{in_features}, out_features_{out_features} {
    if (in_features_ <= 0 || out_features_ <= 0) {
        throw std::invalid_argument{"Linear requires positive in/out features"};
    }
    initialize_linear_params(weight_t_, bias_, in_features_, out_features_, init_type, bias_init, rng);
    
    // Initialize gradients
    grad_weight_ = Tensor::zeros_like(weight_t_);
    grad_bias_ = Tensor::zeros_like(bias_);
}

Tensor Linear::forward(const Tensor& x) const {
    ensure_2d(x);
    if (x.size(1) != in_features_) {
        throw std::runtime_error("Linear::forward input feature mismatch");
    }
    const int m = static_cast<int>(x.size(0));
    const int k = static_cast<int>(x.size(1));
    const int n = out_features_;

    Tensor out({m, n});
    auto* A = const_cast<double*>(x.data_ptr<double>());
    auto* B = const_cast<double*>(weight_t_.data_ptr<double>());
    auto* C = out.data_ptr<double>();

    // C = A * B
    (void)gemm(reinterpret_cast<TYPE_VAL*>(A), reinterpret_cast<TYPE_VAL*>(B), reinterpret_cast<TYPE_VAL*>(C), m, k, n);

    // Add bias (row-major)
    const double* b = bias_.data_ptr<double>();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] += b[j];
        }
    }

    return out;
}

Tensor Linear::backward(const Tensor& grad_output, const Tensor& input) {
    // grad_output: [batch, out_features]
    // input: [batch, in_features]
    
    // grad_input = grad_output @ weight.T
    // grad_weight = input.T @ grad_output
    // grad_bias = sum(grad_output, dim=0)

    // 1. grad_input
    Tensor grad_input = grad_output.matmul(weight_t_.transpose(0, 1));

    // 2. grad_weight
    Tensor dw = input.transpose(0, 1).matmul(grad_output);
    // Accumulate gradients? Or overwrite? Standard is accumulate (zero_grad clears it).
    // But here we might want to just set it if we assume zero_grad was called.
    // Let's accumulate.
    // Check shapes
    if (grad_weight_.numel() != dw.numel()) {
        grad_weight_ = dw; // First time or shape mismatch (shouldn't happen)
    } else {
        grad_weight_ = grad_weight_ + dw;
    }

    // 3. grad_bias
    Tensor db = grad_output.sum(0);
    if (grad_bias_.numel() != db.numel()) {
        grad_bias_ = db;
    } else {
        grad_bias_ = grad_bias_ + db;
    }

    return grad_input;
}

Fnn::Fnn(std::vector<int> layer_sizes,
         ActivationFn activation,
         ActivationFn activation_deriv,
         InitType init_type,
         Scalar bias_init,
         int64_t seed,
         ArchitectureConfig arch_config)
    : layer_sizes_{std::move(layer_sizes)},
      activation_{std::move(activation)},
      activation_deriv_{std::move(activation_deriv)},
      init_type_{init_type},
      bias_init_{bias_init},
      arch_config_{std::move(arch_config)},
      architecture_{arch_config_.architecture} {
    if (layer_sizes_.size() < 2) {
        throw std::invalid_argument{"Network requires at least input and output layers"};
    }
    input_dim_ = layer_sizes_.front();
    output_dim_ = layer_sizes_.back();

    if (architecture_ != NetworkArchitecture::kFnn) {
        throw std::runtime_error("Pure-C backend currently supports only plain FNN architecture");
    }

    pinn::core::Rng rng(seed >= 0 ? static_cast<uint64_t>(seed) : 0x9e3779b97f4a7c15ULL);

    layers_.clear();
    layers_.reserve(layer_sizes_.size() - 1);
    for (size_t i = 0; i + 1 < layer_sizes_.size(); ++i) {
        layers_.emplace_back(layer_sizes_[i], layer_sizes_[i + 1], init_type_, bias_init_, rng);
    }
}

Fnn::Fnn(std::vector<int> layer_sizes,
         const std::string& activation_name,
         InitType init_type,
         Scalar bias_init,
         int64_t seed,
         ArchitectureConfig arch_config)
    : Fnn(layer_sizes,
          activation_from_string(activation_name),
          activation_derivative_from_string(activation_name),
          init_type,
          bias_init,
          seed,
          arch_config) {}

Tensor Fnn::forward(const Tensor& x) {
    ensure_2d(x);
    if (x.size(1) != input_dim_) {
        throw std::runtime_error("Fnn::forward input feature mismatch");
    }

    inputs_.clear();
    pre_activations_.clear(); // Not used yet but good practice

    Tensor out = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
        inputs_.push_back(out); // Cache input to layer i
        out = layers_[i].forward(out);
        if (i + 1 != layers_.size()) {
            // pre_activations_.push_back(out); 
            out = activation_(out);
        }
    }
    return out;
}

Tensor Fnn::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;
    
    // Iterate backwards
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        // If not last layer, backprop through activation
        if (i + 1 != static_cast<int>(layers_.size())) {
            // Derivative of activation
            // We need the input to the activation function.
            // The input to activation was the output of layer[i].forward().
            // We didn't cache it explicitly, but we can recompute or cache it.
            // Let's use the cached input to layer[i+1] which IS the output of activation(layer[i]).
            // Wait, activation input is linear output.
            // activation output is input to next layer.
            // So inputs_[i+1] is activation(linear_out).
            // If we use Tanh, derivative is 1 - y^2. So we can use inputs_[i+1].
            // If we use ReLU, we need x.
            // To be safe and general, we should cache pre-activation.
            // Let's recompute it for now to save memory? No, cache is better.
            // I'll modify forward to cache pre-activations?
            // Actually, let's just recompute linear forward for now? No, that's slow.
            // Let's assume Tanh for now or use the fact that we have inputs_[i+1] which is y.
            // But `activation_deriv_` might expect x.
            // My `make_tanh_deriv` expects x (input to tanh).
            // So I need the output of layer[i].
            // Let's recompute it: layer[i].forward(inputs_[i]).
            Tensor linear_out = layers_[i].forward(inputs_[i]);
            Tensor d_act = activation_deriv_(linear_out);
            grad = grad * d_act; // Element-wise mul
        }
        
        // Backprop through linear
        grad = layers_[i].backward(grad, inputs_[i]);
    }
    return grad;
}

}  // namespace pinn::nn
