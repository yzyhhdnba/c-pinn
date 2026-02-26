#include "pinn/nn/tanh.hpp"

#include <cmath>

namespace pinn::nn {

int tanh(TYPE_VAL* features, int num_nodes, int feature_dim) {
    const int n = num_nodes * feature_dim;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        features[i] = static_cast<TYPE_VAL>(std::tanh(static_cast<double>(features[i])));
    }
    return 0;
}

int tanh_backward(TYPE_VAL* delta, const TYPE_VAL* node_features, int num_nodes, int feature_dim) {
    const int n = num_nodes * feature_dim;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        const TYPE_VAL y = node_features[i];
        delta[i] *= static_cast<TYPE_VAL>(1.0) - y * y;
    }
    return 0;
}

}  // namespace pinn::nn
