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

Fnn::Fnn(std::vector<int> layer_sizes,
         ActivationFn activation,
         InitType init_type,
         Scalar bias_init,
         int64_t seed,
         ArchitectureConfig arch_config)
    : layer_sizes_{std::move(layer_sizes)},
      activation_{std::move(activation)},
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

Tensor Fnn::forward(const Tensor& x) const {
    ensure_2d(x);
    if (x.size(1) != input_dim_) {
        throw std::runtime_error("Fnn::forward input feature mismatch");
    }

    Tensor out = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
        out = layers_[i].forward(out);
        if (i + 1 != layers_.size()) {
            out = activation_(out);
        }
    }
    return out;
}

}  // namespace pinn::nn
