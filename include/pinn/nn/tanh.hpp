#pragma once

#include "pinn/nn/gemm.hpp"  // TYPE_VAL

namespace pinn::nn {

// In-place tanh forward: features[num_nodes * feature_dim]
int tanh(TYPE_VAL* features, int num_nodes, int feature_dim);

// In-place tanh backward using tanh output (node_features already tanh(x)):
// delta[num_nodes * feature_dim] *= (1 - node_features^2)
int tanh_backward(TYPE_VAL* delta, const TYPE_VAL* node_features, int num_nodes, int feature_dim);

}  // namespace pinn::nn
