#include "pinn/nn/flatten.hpp"

#include <stdexcept>

namespace pinn::nn {

FlatLayout make_flat_layout(const std::vector<int>& dims) {
    if (dims.size() < 2) {
        throw std::invalid_argument{"make_flat_layout requires at least 2 dims"};
    }
    FlatLayout layout;
    layout.layer_dims = dims;
    layout.num_layers = static_cast<int>(dims.size()) - 1;
    layout.w_offsets.resize(static_cast<size_t>(layout.num_layers));
    layout.b_offsets.resize(static_cast<size_t>(layout.num_layers));

    int w_off = 0;
    int b_off = 0;
    for (int l = 0; l < layout.num_layers; ++l) {
        const int in = dims[static_cast<size_t>(l)];
        const int out = dims[static_cast<size_t>(l + 1)];
        if (in <= 0 || out <= 0) {
            throw std::invalid_argument{"make_flat_layout: dims must be positive"};
        }
        layout.w_offsets[static_cast<size_t>(l)] = w_off;
        layout.b_offsets[static_cast<size_t>(l)] = b_off;
        w_off += in * out;
        b_off += out;
    }
    layout.size_w = w_off;
    layout.size_b = b_off;
    return layout;
}

static void check_layout_matches(const Fnn& net, const FlatLayout& layout) {
    if (layout.num_layers != net.num_layers()) {
        throw std::runtime_error("flat layout num_layers mismatch");
    }
    if (layout.layer_dims.size() != net.layer_sizes().size()) {
        throw std::runtime_error("flat layout dims length mismatch");
    }
    for (size_t i = 0; i < layout.layer_dims.size(); ++i) {
        if (layout.layer_dims[i] != net.layer_sizes()[i]) {
            throw std::runtime_error("flat layout dims mismatch");
        }
    }
}

int pack_params(const Fnn& net, TYPE_VAL* weights, TYPE_VAL* bias, const FlatLayout& layout) {
    if (!weights || !bias) {
        return -1;
    }
    try {
        check_layout_matches(net, layout);
    } catch (...) {
        return -2;
    }

    for (int l = 0; l < layout.num_layers; ++l) {
        const int in = layout.layer_dims[static_cast<size_t>(l)];
        const int out = layout.layer_dims[static_cast<size_t>(l + 1)];
        const int w_off = layout.w_offsets[static_cast<size_t>(l)];
        const int b_off = layout.b_offsets[static_cast<size_t>(l)];

        const Linear& layer = net.layer(l);
        if (layer.in_features() != in || layer.out_features() != out) {
            return -3;
        }

        const TYPE_VAL* w_src = layer.weight_data();
        const TYPE_VAL* b_src = layer.bias_data();

        for (int i = 0; i < in * out; ++i) {
            weights[w_off + i] = w_src[i];
        }
        for (int i = 0; i < out; ++i) {
            bias[b_off + i] = b_src[i];
        }
    }

    return 0;
}

int unpack_params(Fnn& net, const TYPE_VAL* weights, const TYPE_VAL* bias, const FlatLayout& layout) {
    if (!weights || !bias) {
        return -1;
    }
    try {
        check_layout_matches(net, layout);
    } catch (...) {
        return -2;
    }

    for (int l = 0; l < layout.num_layers; ++l) {
        const int in = layout.layer_dims[static_cast<size_t>(l)];
        const int out = layout.layer_dims[static_cast<size_t>(l + 1)];
        const int w_off = layout.w_offsets[static_cast<size_t>(l)];
        const int b_off = layout.b_offsets[static_cast<size_t>(l)];

        Linear& layer = net.layer(l);
        if (layer.in_features() != in || layer.out_features() != out) {
            return -3;
        }

        TYPE_VAL* w_dst = layer.weight_data();
        TYPE_VAL* b_dst = layer.bias_data();

        for (int i = 0; i < in * out; ++i) {
            w_dst[i] = weights[w_off + i];
        }
        for (int i = 0; i < out; ++i) {
            b_dst[i] = bias[b_off + i];
        }
    }

    return 0;
}

}  // namespace pinn::nn
