#pragma once

#include <vector>

#include "pinn/nn/fnn.hpp"
#include "pinn/nn/gemm.hpp"  // TYPE_VAL

namespace pinn::nn {

struct FlatLayout {
    int num_layers{0};
    std::vector<int> layer_dims;      // size=num_layers+1
    std::vector<int> w_offsets;       // size=num_layers
    std::vector<int> b_offsets;       // size=num_layers
    int size_w{0};
    int size_b{0};
};

FlatLayout make_flat_layout(const std::vector<int>& layer_dims);

// Copy parameters between network tensors and flat arrays.
int pack_params(const Fnn& net, TYPE_VAL* weights, TYPE_VAL* bias, const FlatLayout& layout);
int unpack_params(Fnn& net, const TYPE_VAL* weights, const TYPE_VAL* bias, const FlatLayout& layout);

}  // namespace pinn::nn
