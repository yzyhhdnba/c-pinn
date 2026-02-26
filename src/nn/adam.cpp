#include "pinn/nn/adam.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace pinn::nn {

void adam_param_sizes(int* out_size_w, int* out_size_b, int num_layers, const int* layer_dims) {
    if (num_layers <= 0 || layer_dims == nullptr) {
        throw std::invalid_argument{"adam_param_sizes: invalid args"};
    }
    int size_w = 0;
    int size_b = 0;
    for (int l = 0; l < num_layers; ++l) {
        const int in = layer_dims[l];
        const int out = layer_dims[l + 1];
        if (in <= 0 || out <= 0) {
            throw std::invalid_argument{"adam_param_sizes: layer dims must be positive"};
        }
        size_w += in * out;
        size_b += out;
    }
    if (out_size_w) {
        *out_size_w = size_w;
    }
    if (out_size_b) {
        *out_size_b = size_b;
    }
}

int init_adam(TYPE_VAL** m_w, TYPE_VAL** v_w,
              TYPE_VAL** m_b, TYPE_VAL** v_b,
              int num_layers, const int* layer_dims) {
    if (!m_w || !v_w || !m_b || !v_b) {
        return -1;
    }

    int malloc_size_w = 0;
    int malloc_size_b = 0;
    try {
        adam_param_sizes(&malloc_size_w, &malloc_size_b, num_layers, layer_dims);
    } catch (...) {
        return -2;
    }

    TYPE_VAL* mw = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(malloc_size_w) * sizeof(TYPE_VAL)));
    TYPE_VAL* vw = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(malloc_size_w) * sizeof(TYPE_VAL)));
    TYPE_VAL* mb = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(malloc_size_b) * sizeof(TYPE_VAL)));
    TYPE_VAL* vb = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(malloc_size_b) * sizeof(TYPE_VAL)));

    if (!mw || !vw || !mb || !vb) {
        std::free(mw);
        std::free(vw);
        std::free(mb);
        std::free(vb);
        return -3;
    }

    std::memset(mw, 0, static_cast<size_t>(malloc_size_w) * sizeof(TYPE_VAL));
    std::memset(vw, 0, static_cast<size_t>(malloc_size_w) * sizeof(TYPE_VAL));
    std::memset(mb, 0, static_cast<size_t>(malloc_size_b) * sizeof(TYPE_VAL));
    std::memset(vb, 0, static_cast<size_t>(malloc_size_b) * sizeof(TYPE_VAL));

    *m_w = mw;
    *v_w = vw;
    *m_b = mb;
    *v_b = vb;

    return 0;
}

int gcn_update_adam(TYPE_VAL* weights, TYPE_VAL* bias,
                    TYPE_VAL* grads_w, TYPE_VAL* grads_b,
                    TYPE_VAL* m_w, TYPE_VAL* v_w,
                    TYPE_VAL* m_b, TYPE_VAL* v_b,
                    int num_layers, const int* layer_dims,
                    float lr, int t,
                    float beta1, float beta2, float epsilon,
                    float weight_decay) {
    if (!weights || !bias || !grads_w || !grads_b || !m_w || !v_w || !m_b || !v_b || !layer_dims) {
        return -1;
    }
    if (num_layers <= 0 || t <= 0) {
        return -2;
    }

    int size_w = 0;
    int size_b = 0;
    try {
        adam_param_sizes(&size_w, &size_b, num_layers, layer_dims);
    } catch (...) {
        return -3;
    }

    const TYPE_VAL b1 = static_cast<TYPE_VAL>(beta1);
    const TYPE_VAL b2 = static_cast<TYPE_VAL>(beta2);
    const TYPE_VAL one = static_cast<TYPE_VAL>(1.0);
    const TYPE_VAL lr_d = static_cast<TYPE_VAL>(lr);
    const TYPE_VAL eps = static_cast<TYPE_VAL>(epsilon);
    const TYPE_VAL wd = static_cast<TYPE_VAL>(weight_decay);

    const TYPE_VAL b1t = static_cast<TYPE_VAL>(std::pow(beta1, t));
    const TYPE_VAL b2t = static_cast<TYPE_VAL>(std::pow(beta2, t));
    const TYPE_VAL inv_1mb1t = one / (one - b1t);
    const TYPE_VAL inv_1mb2t = one / (one - b2t);

    // Update weights
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < size_w; ++i) {
        TYPE_VAL g = grads_w[i];
        // L2 regularization
        if (wd != static_cast<TYPE_VAL>(0.0)) {
            g += wd * weights[i];
            grads_w[i] = g;
        }

        TYPE_VAL mw = m_w[i] = b1 * m_w[i] + (one - b1) * g;
        TYPE_VAL vw = v_w[i] = b2 * v_w[i] + (one - b2) * g * g;

        const TYPE_VAL m_hat = mw * inv_1mb1t;
        const TYPE_VAL v_hat = vw * inv_1mb2t;

        weights[i] -= lr_d * m_hat / (std::sqrt(v_hat) + eps);
    }

    // Update bias (no weight decay by default)
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < size_b; ++i) {
        const TYPE_VAL g = grads_b[i];
        TYPE_VAL mb = m_b[i] = b1 * m_b[i] + (one - b1) * g;
        TYPE_VAL vb = v_b[i] = b2 * v_b[i] + (one - b2) * g * g;

        const TYPE_VAL m_hat = mb * inv_1mb1t;
        const TYPE_VAL v_hat = vb * inv_1mb2t;

        bias[i] -= lr_d * m_hat / (std::sqrt(v_hat) + eps);
    }

    return 0;
}

}  // namespace pinn::nn
