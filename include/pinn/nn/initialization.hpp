#pragma once

#include <string>

#include "pinn/types.hpp"
#include "pinn/core/rng.hpp"
#include "pinn/nn/gemm.hpp"  // TYPE_VAL

namespace pinn::nn {

enum class InitType {
    kXavierUniform,
    kXavierNormal,
    kHeUniform,
    kHeNormal,
    kZero
};

InitType init_from_string(const std::string& name);

// Initialize a linear layer weight (stored as weight_t: [fan_in, fan_out]) and bias: [fan_out].
// Strict C-style: fill flat buffers with explicit loops.
void initialize_linear_params(Tensor& weight_t,
                              Tensor& bias,
                              int fan_in,
                              int fan_out,
                              InitType init_type,
                              Scalar bias_init,
                              pinn::core::Rng& rng);

// Pure C-style Xavier/Glorot uniform initializer using libc rand()/srand(0) and U(-a,a).
// input_weights[i] must point to a contiguous row-major [layer_dims[i] * layer_dims[i+1]] buffer.
// Returns 0 on success, non-zero on invalid args.
int init_weights_xavier_uniform(TYPE_VAL** input_weights, int num_layers, int* layer_dims);

// Pure C-style Kaiming/He uniform initializer using libc rand()/srand(0) and U(-a,a).
// For ReLU-family activations, a = sqrt(6 / fan_in).
// input_weights[i] must point to a contiguous row-major [layer_dims[i] * layer_dims[i+1]] buffer.
// Returns 0 on success, non-zero on invalid args.
int init_weights_kaiming_uniform(TYPE_VAL** input_weights, int num_layers, int* layer_dims);

}  // namespace pinn::nn
