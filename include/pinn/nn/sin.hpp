#pragma once

#include "pinn/nn/gemm.hpp"  // TYPE_VAL

namespace pinn::nn {

// In-place sin forward: features[num_nodes * feature_dim]
int sin(TYPE_VAL* features, int num_nodes, int feature_dim);

// In-place sin backward using pre-activation input (node_inputs are x before sin):
// delta[num_nodes * feature_dim] *= cos(node_inputs)
int sin_backward(TYPE_VAL* delta, const TYPE_VAL* node_inputs, int num_nodes, int feature_dim);

}  // namespace pinn::nn
