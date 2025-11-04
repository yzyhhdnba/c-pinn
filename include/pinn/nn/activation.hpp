#pragma once

#include <functional>
#include <string>

#include "pinn/types.hpp"

namespace pinn::nn {

using ActivationFn = std::function<Tensor(const Tensor&)>;

ActivationFn activation_from_string(const std::string& name);
std::string normalize_activation_name(const std::string& name);

}  // namespace pinn::nn
