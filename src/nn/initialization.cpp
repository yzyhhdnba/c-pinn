#include "pinn/nn/initialization.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include <torch/torch.h>

namespace pinn::nn {

namespace {
std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

void initialize_linear(torch::nn::LinearImpl& linear, InitType init_type, torch::optional<Scalar> bias) {
    auto& weight = linear.weight;
    switch (init_type) {
        case InitType::kXavierUniform:
            torch::nn::init::xavier_uniform_(weight);
            break;
        case InitType::kXavierNormal:
            torch::nn::init::xavier_normal_(weight);
            break;
        case InitType::kHeUniform:
            torch::nn::init::kaiming_uniform_(weight, std::sqrt(5.0));
            break;
        case InitType::kHeNormal:
            torch::nn::init::kaiming_normal_(weight, std::sqrt(5.0));
            break;
        case InitType::kZero:
            torch::nn::init::constant_(weight, 0.0);
            break;
    }
    if (linear.bias.defined()) {
        if (bias.has_value()) {
            torch::nn::init::constant_(linear.bias, *bias);
        } else {
            torch::nn::init::zeros_(linear.bias);
        }
    }
}

}  // namespace

InitType init_from_string(const std::string& name) {
    const auto lower = to_lower(name);
    if (lower == "xavier_uniform") {
        return InitType::kXavierUniform;
    }
    if (lower == "xavier_normal") {
        return InitType::kXavierNormal;
    }
    if (lower == "he_uniform" || lower == "kaiming_uniform") {
        return InitType::kHeUniform;
    }
    if (lower == "he_normal" || lower == "kaiming_normal") {
        return InitType::kHeNormal;
    }
    if (lower == "zero") {
        return InitType::kZero;
    }
    throw std::invalid_argument{"Unsupported initialization: " + name};
}

void apply_initialization(torch::nn::Module& module, InitType init_type, torch::optional<Scalar> bias) {
    module.apply([&](torch::nn::Module& m) {
        if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(&m)) {
            initialize_linear(*linear, init_type, bias);
        }
    });
}

}  // namespace pinn::nn
