#include "pinn/nn/activation.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace pinn::nn {

namespace {
std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

ActivationFn make_relu() {
    return [](const Tensor& x) { return x.relu(); };
}

ActivationFn make_tanh() {
    return [](const Tensor& x) { return x.tanh(); };
}

ActivationFn make_sin() {
    return [](const Tensor& x) { return x.sin(); };
}

ActivationFn make_relu_deriv() {
    return [](const Tensor& x) {
        // x > 0 ? 1 : 0
        // We don't have a "gt" operator returning a mask yet.
        // Let's implement a simple loop or add "step" to Tensor.
        // For now, manual loop.
        Tensor out = Tensor::zeros_like(x);
        const double* in_ptr = x.data_ptr<double>();
        double* out_ptr = out.data_ptr<double>();
        int64_t n = x.numel();
        for(int64_t i=0; i<n; ++i) {
            out_ptr[i] = in_ptr[i] > 0.0 ? 1.0 : 0.0;
        }
        return out;
    };
}

ActivationFn make_tanh_deriv() {
    return [](const Tensor& x) {
        // 1 - tanh^2(x)
        Tensor t = x.tanh();
        return Tensor::ones_like(x) - t * t;
    };
}

ActivationFn make_sin_deriv() {
    return [](const Tensor& x) {
        // cos(x)
        // We don't have cos in Tensor yet. Need to add it or loop.
        // Let's add cos to Tensor later or loop here.
        Tensor out = Tensor::zeros_like(x);
        const double* in_ptr = x.data_ptr<double>();
        double* out_ptr = out.data_ptr<double>();
        int64_t n = x.numel();
        for(int64_t i=0; i<n; ++i) {
            out_ptr[i] = std::cos(in_ptr[i]);
        }
        return out;
    };
}

}  // namespace

std::string normalize_activation_name(const std::string& name) {
    return to_lower(name);
}

ActivationFn activation_from_string(const std::string& name) {
    const auto lower = to_lower(name);
    if (lower == "relu") {
        return make_relu();
    }
    if (lower == "tanh") {
        return make_tanh();
    }
    if (lower == "sin" || lower == "sine") {
        return make_sin();
    }
    throw std::invalid_argument{"Unsupported activation: " + name};
}

ActivationFn activation_derivative_from_string(const std::string& name) {
    const auto lower = to_lower(name);
    if (lower == "relu") {
        return make_relu_deriv();
    }
    if (lower == "tanh") {
        return make_tanh_deriv();
    }
    if (lower == "sin" || lower == "sine") {
        return make_sin_deriv();
    }
    throw std::invalid_argument{"Unsupported activation: " + name};
}

}  // namespace pinn::nn
