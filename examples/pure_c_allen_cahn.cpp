#include "pinn/nn/fnn.hpp"
#include "pinn/core/tensor.hpp"
#include "pinn/nn/optimizer.hpp"
#include "pinn/utils/stencil.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace pinn;

int main() {
    std::cout << "=== Pure C Allen-Cahn Solver (Stencil Framework) ===" << std::endl;

    // 1. Setup Network
    std::vector<int> layers = {2, 50, 50, 50, 1};
    nn::Fnn net(layers, "tanh", nn::InitType::kXavierUniform, 0.0, 42);

    // 2. Setup Optimizer
    nn::AdamOptimizer::Options opts;
    opts.lr = 1e-3;
    nn::AdamOptimizer optimizer(net, opts);

    // 3. Training Loop
    const int iterations = 1000;
    const int batch_size = 64;
    const double h = 1e-3;  // Finite difference step size

    for (int iter = 0; iter < iterations; ++iter) {
        optimizer.zero_grad();

        // Sample batch (x, t) in domain [0, 1] x [0, 1]
        auto x_t = core::Tensor::rand_uniform({batch_size, 2});

        // ============================================================
        // Define stencil points for Allen-Cahn:
        // u_t - 0.0001*u_xx + 5(u^3 - u) = 0
        //
        // u_t  = (u(t+h) - u(t-h)) / 2h
        // u_xx = (u(x+h) - 2u(x) + u(x-h)) / h^2
        // ============================================================

        // Build stencil points: [center, x+h, x-h, t+h, t-h]
        std::vector<StencilPoint> stencils;

        // Center point (x, t)
        stencils.emplace_back(x_t.clone(), 0.0);

        // x+h, x-h (for u_xx)
        auto make_x_shift = [&](double shift) {
            core::Tensor p = x_t.clone();
            double* d = p.data_ptr<double>();
            for (int i = 0; i < batch_size; ++i) d[i*2 + 0] += shift;
            return p;
        };
        stencils.emplace_back(make_x_shift(h), 0.0);
        stencils.emplace_back(make_x_shift(-h), 0.0);

        // t+h, t-h (for u_t)
        auto make_t_shift = [&](double shift) {
            core::Tensor p = x_t.clone();
            double* d = p.data_ptr<double>();
            for (int i = 0; i < batch_size; ++i) d[i*2 + 1] += shift;
            return p;
        };
        stencils.emplace_back(make_t_shift(h), 0.0);
        stencils.emplace_back(make_t_shift(-h), 0.0);

        // Forward pass through all stencil points
        forward_all(net, stencils);

        // Extract outputs by index
        // 0: center, 1: x+h, 2: x-h, 3: t+h, 4: t-h
        auto& u_c   = stencils[0].output;
        auto& u_x_p = stencils[1].output;
        auto& u_x_m = stencils[2].output;
        auto& u_t_p = stencils[3].output;
        auto& u_t_m = stencils[4].output;

        // ============================================================
        // Compute PDE residual: R = u_t - 0.0001*u_xx + 5(u^3 - u)
        // ============================================================

        double inv_2h = 1.0 / (2.0 * h);
        double inv_h2 = 1.0 / (h * h);
        double diff_coeff = 0.0001;

        // u_t ~ (u(t+h) - u(t-h)) / 2h
        auto u_t = (u_t_p - u_t_m) * inv_2h;

        // u_xx ~ (u(x+h) - 2u(x) + u(x-h)) / h^2
        auto u_xx = (u_x_p - u_c * 2.0 + u_x_m) * inv_h2;

        // Reaction term: 5(u^3 - u) = 5u^3 - 5u
        auto u_cubed = u_c * u_c * u_c;
        auto R = u_t - u_xx * diff_coeff + u_cubed * 5.0 - u_c * 5.0;

        auto loss = R.pow(2.0).mean_all();

        if (iter % 100 == 0) {
            std::cout << "Iter " << iter << " Loss: " << loss.item<double>() << std::endl;
        }

        // ============================================================
        // Compute gradients for each stencil point (chain rule)
        // L = mean(R^2), dL/dR = 2R/N
        // ============================================================

        double N = static_cast<double>(batch_size);
        auto dL_dR = R * (2.0 / N);

        // R = u_t - diff*u_xx + 5u^3 - 5u
        // u_t = (u_tp - u_tm)/2h
        // u_xx = (u_xp - 2u_c + u_xm)/h^2
        //
        // dR/du_tp = 1/2h
        // dR/du_tm = -1/2h
        // dR/du_xp = -diff/h^2
        // dR/du_xm = -diff/h^2
        // dR/du_c = -diff * (-2/h^2) + 15u_c^2 - 5 = 2*diff/h^2 + 15u_c^2 - 5

        auto grad_u_tp = dL_dR * inv_2h;
        auto grad_u_tm = dL_dR * (-inv_2h);
        auto grad_u_xp = dL_dR * (-diff_coeff * inv_h2);
        auto grad_u_xm = dL_dR * (-diff_coeff * inv_h2);

        // dR/du_c = 2*diff/h^2 + 15u_c^2 - 5
        auto term_const = dL_dR * (2.0 * diff_coeff * inv_h2);
        auto term_nonlinear = dL_dR * (u_c * u_c * 15.0 + (-5.0));
        auto grad_u_c = term_const + term_nonlinear;

        // ============================================================
        // Backward pass through all stencil points using for loop
        // ============================================================

        net.forward(stencils[0].input); net.backward(grad_u_c);
        net.forward(stencils[1].input); net.backward(grad_u_xp);
        net.forward(stencils[2].input); net.backward(grad_u_xm);
        net.forward(stencils[3].input); net.backward(grad_u_tp);
        net.forward(stencils[4].input); net.backward(grad_u_tm);

        optimizer.step();
    }

    std::cout << "Training finished." << std::endl;
    return 0;
}
