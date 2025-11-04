#include "pinn/nn/activation.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <torch/torch.h>

namespace pinn::nn {

namespace {
std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

ActivationFn make_relu() {
    return [](const Tensor& x) { return torch::relu(x); };
}

ActivationFn make_tanh() {
    return [](const Tensor& x) { return torch::tanh(x); };
}

ActivationFn make_sin() {
    return [](const Tensor& x) { return torch::sin(x); };
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

}  // namespace pinn::nn
