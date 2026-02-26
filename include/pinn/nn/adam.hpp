#pragma once

#include "pinn/nn/gemm.hpp"  // TYPE_VAL

namespace pinn::nn {

// Compute total parameter sizes for a dense MLP described by layer_dims.
// layer_dims has length (num_layers + 1): [in, h1, ..., out]
void adam_param_sizes(int* out_size_w, int* out_size_b, int num_layers, const int* layer_dims);

// Allocate and zero Adam state buffers.
// Caller must free() the returned pointers.
int init_adam(TYPE_VAL** m_w, TYPE_VAL** v_w,
              TYPE_VAL** m_b, TYPE_VAL** v_b,
              int num_layers, const int* layer_dims);

// In-place Adam update for dense MLP weights/bias stored as flat arrays.
// weights: concatenated per-layer matrices (row-major): [layer_dims[l] * layer_dims[l+1]]
// bias: concatenated per-layer vectors: [layer_dims[l+1]]
// grads_w/grads_b correspond to weights/bias.
int gcn_update_adam(TYPE_VAL* weights, TYPE_VAL* bias,
                    TYPE_VAL* grads_w, TYPE_VAL* grads_b,
                    TYPE_VAL* m_w, TYPE_VAL* v_w,
                    TYPE_VAL* m_b, TYPE_VAL* v_b,
                    int num_layers, const int* layer_dims,
                    float lr, int t,
                    float beta1, float beta2, float epsilon,
                    float weight_decay);

}  // namespace pinn::nn
