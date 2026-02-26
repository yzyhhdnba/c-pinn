#include "pinn/nn/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "pinn/nn/adam.hpp"
#include "pinn/nn/lbfgs.hpp"

namespace pinn::nn {

// ============================================================================
// AdamOptimizer
// ============================================================================

AdamOptimizer::AdamOptimizer(Fnn& network, Options options)
    : network_{network}, options_{options} {
    num_layers_ = network_.num_layers();
    if (num_layers_ == 0) {
        throw std::runtime_error("AdamOptimizer: network has no layers");
    }

    layer_dims_.resize(num_layers_ + 1);
    layer_dims_[0] = network_.input_dim();
    for (int i = 0; i < num_layers_; ++i) {
        layer_dims_[i + 1] = network_.layer(i).out_features();
    }

    int size_w = 0, size_b = 0;
    adam_param_sizes(&size_w, &size_b, num_layers_, layer_dims_.data());

    // Allocate state
    int status = init_adam(&m_w_, &v_w_, &m_b_, &v_b_, num_layers_, layer_dims_.data());
    if (status != 0) {
        throw std::runtime_error("AdamOptimizer: init_adam failed");
    }

    // Allocate gradient buffers
    grads_w_ = static_cast<TYPE_VAL*>(std::malloc(size_w * sizeof(TYPE_VAL)));
    grads_b_ = static_cast<TYPE_VAL*>(std::malloc(size_b * sizeof(TYPE_VAL)));
    if (!grads_w_ || !grads_b_) {
        throw std::bad_alloc();
    }
    std::memset(grads_w_, 0, size_w * sizeof(TYPE_VAL));
    std::memset(grads_b_, 0, size_b * sizeof(TYPE_VAL));
}

AdamOptimizer::~AdamOptimizer() {
    std::free(m_w_);
    std::free(v_w_);
    std::free(m_b_);
    std::free(v_b_);
    std::free(grads_w_);
    std::free(grads_b_);
}

void AdamOptimizer::step() {
    t_++;
    const double lr = static_cast<double>(options_.lr);
    const double beta1 = static_cast<double>(options_.beta1);
    const double beta2 = static_cast<double>(options_.beta2);
    const double epsilon = static_cast<double>(options_.epsilon);
    const double weight_decay = static_cast<double>(options_.weight_decay);

    // Bias correction
    const double bias_correction1 = 1.0 - std::pow(beta1, t_);
    const double bias_correction2 = 1.0 - std::pow(beta2, t_);

    // Iterate over all layers
    int param_offset_w = 0;
    int param_offset_b = 0;

    for (int i = 0; i < network_.num_layers(); ++i) {
        auto& layer = network_.layer(i);
        
        // Weights
        {
            Tensor& w = layer.weight_t();
            Tensor& g = layer.grad_weight();
            double* w_ptr = w.data_ptr<double>();
            double* g_ptr = g.data_ptr<double>();
            int64_t size = w.numel();

            // Check if gradient is defined (might be first step)
            if (!g.defined() || g.numel() != size) {
                // Skip update if no gradient
            } else {
                for (int64_t j = 0; j < size; ++j) {
                    double grad = g_ptr[j];
                    if (weight_decay != 0.0) {
                        grad += weight_decay * w_ptr[j];
                    }

                    // Update moments
                    m_w_[param_offset_w + j] = beta1 * m_w_[param_offset_w + j] + (1.0 - beta1) * grad;
                    v_w_[param_offset_w + j] = beta2 * v_w_[param_offset_w + j] + (1.0 - beta2) * grad * grad;

                    double m_hat = m_w_[param_offset_w + j] / bias_correction1;
                    double v_hat = v_w_[param_offset_w + j] / bias_correction2;

                    w_ptr[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }
            param_offset_w += size;
        }

        // Biases
        {
            Tensor& b = layer.bias();
            Tensor& g = layer.grad_bias();
            double* b_ptr = b.data_ptr<double>();
            double* g_ptr = g.data_ptr<double>();
            int64_t size = b.numel();

            if (!g.defined() || g.numel() != size) {
                // Skip
            } else {
                for (int64_t j = 0; j < size; ++j) {
                    double grad = g_ptr[j];
                    
                    m_b_[param_offset_b + j] = beta1 * m_b_[param_offset_b + j] + (1.0 - beta1) * grad;
                    v_b_[param_offset_b + j] = beta2 * v_b_[param_offset_b + j] + (1.0 - beta2) * grad * grad;

                    double m_hat = m_b_[param_offset_b + j] / bias_correction1;
                    double v_hat = v_b_[param_offset_b + j] / bias_correction2;

                    b_ptr[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }
            param_offset_b += size;
        }
    }
}

void AdamOptimizer::zero_grad() {
    for (int i = 0; i < network_.num_layers(); ++i) {
        auto& layer = network_.layer(i);
        // Reset gradients to zero
        if (layer.grad_weight().defined()) {
            // Re-allocate or fill? Fill is faster if size matches.
            // Actually, let's just assign a new zero tensor or fill.
            // Tensor::zeros_like allocates new memory.
            // We can use memset if we expose buffer.
            // Or just assign zeros_like.
            layer.grad_weight() = Tensor::zeros_like(layer.weight_t());
        }
        if (layer.grad_bias().defined()) {
            layer.grad_bias() = Tensor::zeros_like(layer.bias());
        }
    }
}

void AdamOptimizer::set_learning_rate(Scalar lr) {
    options_.lr = lr;
}

Scalar AdamOptimizer::get_learning_rate() const {
    return options_.lr;
}

// ============================================================================
// LbfgsOptimizer
// ============================================================================

LbfgsOptimizer::LbfgsOptimizer(Fnn& network, Options options)
    : network_{network}, options_{options} {
    // L-BFGS implementation is more complex, placeholder for now
}

LbfgsOptimizer::~LbfgsOptimizer() = default;

void LbfgsOptimizer::step() {
    // TODO: Implement L-BFGS step
}

void LbfgsOptimizer::zero_grad() {
    // L-BFGS doesn't accumulate gradients the same way
}

void LbfgsOptimizer::set_learning_rate(Scalar lr) {
    options_.lr = lr;
}

Scalar LbfgsOptimizer::get_learning_rate() const {
    return options_.lr;
}

// ============================================================================
// Gradient clipping
// ============================================================================

void clip_grad_norm_(Fnn& network, Scalar max_norm) {
    if (max_norm <= 0.0) {
        return;
    }

    // TODO: Compute total gradient norm across all parameters
    // For now, this is a placeholder that requires gradient access
    
    // Conceptually:
    // 1. Compute total_norm = sqrt(sum of squared gradients)
    // 2. If total_norm > max_norm, scale all gradients by (max_norm / total_norm)
}

}  // namespace pinn::nn
