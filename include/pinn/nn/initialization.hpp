#pragma once

#include <string>

#include "pinn/types.hpp"

namespace pinn::nn {

enum class InitType {
    kXavierUniform,
    kXavierNormal,
    kHeUniform,
    kHeNormal,
    kZero
};

InitType init_from_string(const std::string& name);
void apply_initialization(torch::nn::Module& module, InitType init_type, torch::optional<Scalar> bias = torch::nullopt);

}  // namespace pinn::nn
