#pragma once

#include "pinn/nn/gemm.hpp"  // TYPE_VAL

namespace pinn::nn {

// In-place ReLU forward: features[num_nodes * feature_dim]
int relu(TYPE_VAL* features, int num_nodes, int feature_dim);

// In-place ReLU backward (mask delta by node_features>0)
// delta[num_nodes * feature_dim] *= (node_features>0 ? 1 : 0)
int relu_backward(TYPE_VAL* delta, const TYPE_VAL* node_features, int num_nodes, int feature_dim);

}  // namespace pinn::nn
