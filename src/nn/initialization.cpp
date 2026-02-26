#include "pinn/nn/initialization.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace pinn::nn {

namespace {
std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

double uniform_symmetric(pinn::core::Rng& rng, double limit) {
    // U(-limit, limit)
    return (rng.uniform01() * 2.0 - 1.0) * limit;
}

double normal_scaled(pinn::core::Rng& rng, double stddev) {
    return rng.normal01() * stddev;
}

}  // namespace

InitType init_from_string(const std::string& name) {
    const auto lower = to_lower(name);
    if (lower == "xavier_uniform") {
        return InitType::kXavierUniform;
    }
    if (lower == "xavier_normal") {
        return InitType::kXavierNormal;
    }
    if (lower == "he_uniform" || lower == "kaiming_uniform") {
        return InitType::kHeUniform;
    }
    if (lower == "he_normal" || lower == "kaiming_normal") {
        return InitType::kHeNormal;
    }
    if (lower == "zero") {
        return InitType::kZero;
    }
    throw std::invalid_argument{"Unsupported initialization: " + name};
}

void initialize_linear_params(Tensor& weight_t,
                              Tensor& bias,
                              int fan_in,
                              int fan_out,
                              InitType init_type,
                              Scalar bias_init,
                              pinn::core::Rng& rng) {
    if (fan_in <= 0 || fan_out <= 0) {
        throw std::invalid_argument{"fan_in/fan_out must be positive"};
    }
    weight_t = Tensor({fan_in, fan_out});
    bias = Tensor({fan_out});

    auto* w = weight_t.data_ptr<double>();
    auto* b = bias.data_ptr<double>();
    const int64_t w_numel = weight_t.numel();

    double limit = 0.0;
    double stddev = 0.0;
    switch (init_type) {
        case InitType::kXavierUniform:
            limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
            for (int64_t i = 0; i < w_numel; ++i) {
                w[i] = uniform_symmetric(rng, limit);
            }
            break;
        case InitType::kXavierNormal:
            stddev = std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
            for (int64_t i = 0; i < w_numel; ++i) {
                w[i] = normal_scaled(rng, stddev);
            }
            break;
        case InitType::kHeUniform:
            limit = std::sqrt(6.0 / static_cast<double>(fan_in));
            for (int64_t i = 0; i < w_numel; ++i) {
                w[i] = uniform_symmetric(rng, limit);
            }
            break;
        case InitType::kHeNormal:
            stddev = std::sqrt(2.0 / static_cast<double>(fan_in));
            for (int64_t i = 0; i < w_numel; ++i) {
                w[i] = normal_scaled(rng, stddev);
            }
            break;
        case InitType::kZero:
            for (int64_t i = 0; i < w_numel; ++i) {
                w[i] = 0.0;
            }
            break;
    }

    for (int i = 0; i < fan_out; ++i) {
        b[i] = static_cast<double>(bias_init);
    }
}

#define RAND_RANGE(a) ((a) * (2.0 * ((double)rand()) / (RAND_MAX) - 1.0))

int init_weights_xavier_uniform(TYPE_VAL** input_weights, int num_layers, int* layer_dims) {
    if (input_weights == nullptr || layer_dims == nullptr) {
        return 1;
    }
    if (num_layers <= 0) {
        return 2;
    }

    // fixed seed for reproducibility (as requested)
    std::srand(0);

    for (int i = 0; i < num_layers; i++) {
        const int fan_in = layer_dims[i];
        const int fan_out = layer_dims[i + 1];
        if (fan_in <= 0 || fan_out <= 0) {
            return 3;
        }
        if (input_weights[i] == nullptr) {
            return 4;
        }

        const double a = std::sqrt(6.0 / (static_cast<double>(fan_in + fan_out)));
        TYPE_VAL* w = input_weights[i];
        for (int j = 0; j < fan_in; j++) {
            for (int k = 0; k < fan_out; k++) {
                const int idx = j * fan_out + k;
                w[idx] = static_cast<TYPE_VAL>(RAND_RANGE(a));
            }
        }
    }

    return 0;
}

#undef RAND_RANGE

#define RAND_RANGE(a) ((a) * (2.0 * ((double)rand()) / (RAND_MAX) - 1.0))

int init_weights_kaiming_uniform(TYPE_VAL** input_weights, int num_layers, int* layer_dims) {
    if (input_weights == nullptr || layer_dims == nullptr) {
        return 1;
    }
    if (num_layers <= 0) {
        return 2;
    }

    // fixed seed for reproducibility (same behavior as xavier helper)
    std::srand(0);

    for (int i = 0; i < num_layers; i++) {
        const int fan_in = layer_dims[i];
        const int fan_out = layer_dims[i + 1];
        (void)fan_out;
        if (fan_in <= 0) {
            return 3;
        }
        if (layer_dims[i + 1] <= 0) {
            return 3;
        }
        if (input_weights[i] == nullptr) {
            return 4;
        }

        const double a = std::sqrt(6.0 / static_cast<double>(fan_in));
        TYPE_VAL* w = input_weights[i];
        for (int j = 0; j < fan_in; j++) {
            for (int k = 0; k < layer_dims[i + 1]; k++) {
                const int idx = j * layer_dims[i + 1] + k;
                w[idx] = static_cast<TYPE_VAL>(RAND_RANGE(a));
            }
        }
    }

    return 0;
}

#undef RAND_RANGE

}  // namespace pinn::nn
