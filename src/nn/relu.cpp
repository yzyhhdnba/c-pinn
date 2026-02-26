#include "pinn/nn/relu.hpp"

namespace pinn::nn {

int relu(TYPE_VAL* features, int num_nodes, int feature_dim) {
    const int n = num_nodes * feature_dim;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        const TYPE_VAL v = features[i];
        features[i] = v > static_cast<TYPE_VAL>(0.0) ? v : static_cast<TYPE_VAL>(0.0);
    }
    return 0;
}

int relu_backward(TYPE_VAL* delta, const TYPE_VAL* node_features, int num_nodes, int feature_dim) {
    const int n = num_nodes * feature_dim;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        delta[i] *= (node_features[i] > static_cast<TYPE_VAL>(0.0)) ? static_cast<TYPE_VAL>(1.0) : static_cast<TYPE_VAL>(0.0);
    }
    return 0;
}

}  // namespace pinn::nn
