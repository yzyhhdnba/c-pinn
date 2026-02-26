#include "pinn/nn/sin.hpp"

#include <cmath>

namespace pinn::nn {

int sin(TYPE_VAL* features, int num_nodes, int feature_dim) {
    const int n = num_nodes * feature_dim;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        features[i] = static_cast<TYPE_VAL>(std::sin(static_cast<double>(features[i])));
    }
    return 0;
}

int sin_backward(TYPE_VAL* delta, const TYPE_VAL* node_inputs, int num_nodes, int feature_dim) {
    const int n = num_nodes * feature_dim;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        delta[i] *= static_cast<TYPE_VAL>(std::cos(static_cast<double>(node_inputs[i])));
    }
    return 0;
}

}  // namespace pinn::nn
