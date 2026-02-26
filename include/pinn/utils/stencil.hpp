#pragma once

#include <vector>
#include <string>

#include "pinn/core/tensor.hpp"
#include "pinn/nn/fnn.hpp"

namespace pinn {

// Represents a single stencil point with its input tensor and gradient weight
struct StencilPoint {
    core::Tensor input;   // Input coordinates for this stencil point
    core::Tensor output;  // Network output (filled after forward)
    double grad_weight;   // dL/du for this point (how much this point contributes to gradient)

    StencilPoint() : grad_weight(0.0) {}
    StencilPoint(const core::Tensor& inp, double weight) : input(inp), grad_weight(weight) {}
};

// Create a set of stencil points by shifting coordinates
inline std::vector<StencilPoint> make_stencil_points(
    const core::Tensor& base,
    const std::vector<int>& cols,      // which dimensions to shift
    const std::vector<double>& shifts, // shift amounts
    const std::vector<double>& weights,// gradient weights
    int batch_size
) {
    std::vector<StencilPoint> result;
    size_t n = cols.size();
    result.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        core::Tensor shifted = base.clone();
        double* data = shifted.data_ptr<double>();
        for (int b = 0; b < batch_size; ++b) {
            data[b * base.shape()[1] + cols[i]] += shifts[i];
        }
        result.emplace_back(std::move(shifted), weights[i]);
    }

    return result;
}

// Forward pass through all stencil points (stores outputs in place)
inline void forward_all(nn::Fnn& net, std::vector<StencilPoint>& stencils) {
    for (auto& sp : stencils) {
        sp.output = net.forward(sp.input);
    }
}

// Backward pass through all stencil points with their gradient weights
// dL_dR: the gradient of loss with respect to the residual (already scaled by 2/N)
inline void backward_all(nn::Fnn& net, std::vector<StencilPoint>& stencils, const core::Tensor& dL_dR) {
    for (auto& sp : stencils) {
        if (std::abs(sp.grad_weight) < 1e-15) continue;  // skip zero weights
        net.forward(sp.input);
        core::Tensor grad = dL_dR * sp.grad_weight;
        net.backward(grad);
    }
}

// Simplified: forward and backward in one pass (re-runs forward before each backward)
inline void forward_backward_all(nn::Fnn& net, std::vector<StencilPoint>& stencils, const core::Tensor& dL_dR) {
    backward_all(net, stencils, dL_dR);
}

// Helper to create common stencils for a given PDE order
namespace stencil_factory {

// Second-order central difference for first derivative: f'(x) ≈ (f(x+h) - f(x-h)) / 2h
inline StencilPoint first_derivative_plus(int col, double h) {
    return StencilPoint(core::Tensor(), 1.0 / (2.0 * h));  // placeholder, use make_stencil_points instead
}

}  // namespace stencil_factory

}  // namespace pinn
