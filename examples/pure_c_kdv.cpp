#include "pinn/nn/fnn.hpp"
#include "pinn/core/tensor.hpp"
#include "pinn/nn/optimizer.hpp"
#include "pinn/utils/stencil.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace pinn;

int main() {
    std::cout << "=== Pure C KdV Solver (Stencil Framework) ===" << std::endl;

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
        // Define stencil points for KdV: u_t + 6uu_x + u_xxx = 0
        //
        // u_t  = (u(t+h) - u(t-h)) / 2h
        // u_x  = (u(x+h) - u(x-h)) / 2h
        // u_xxx = (u(x+2h) - 2u(x+h) + 2u(x-h) - u(x-2h)) / (2h^3)
        // ============================================================

        // Column indices: 0 = x, 1 = t
        std::vector<int> cols =    {0,  0,  0,   0,   1,   1};
        std::vector<double> shifts = {0,  h, -h,  2*h, -2*h, h, -h};  // 7 points total
        // Actually let's do it properly with all 7 points

        // Build stencil points: [center, x+h, x-h, x+2h, x-2h, t+h, t-h]
        std::vector<StencilPoint> stencils;

        // Center point (x, t)
        stencils.emplace_back(x_t.clone(), 0.0);  // weight set later

        // x+h, x-h, x+2h, x-2h (for u_x and u_xxx)
        auto make_x_shift = [&](double shift) {
            core::Tensor p = x_t.clone();
            double* d = p.data_ptr<double>();
            for (int i = 0; i < batch_size; ++i) d[i*2 + 0] += shift;
            return p;
        };
        stencils.emplace_back(make_x_shift(h), 0.0);
        stencils.emplace_back(make_x_shift(-h), 0.0);
        stencils.emplace_back(make_x_shift(2*h), 0.0);
        stencils.emplace_back(make_x_shift(-2*h), 0.0);

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
        // 0: center, 1: x+h, 2: x-h, 3: x+2h, 4: x-2h, 5: t+h, 6: t-h
        auto& u_c   = stencils[0].output;
        auto& u_x_p = stencils[1].output;
        auto& u_x_m = stencils[2].output;
        auto& u_x_pp = stencils[3].output;
        auto& u_x_mm = stencils[4].output;
        auto& u_t_p = stencils[5].output;
        auto& u_t_m = stencils[6].output;

        // ============================================================
        // Compute PDE residual: R = u_t + 6*u*u_x + u_xxx
        // ============================================================

        double inv_2h = 1.0 / (2.0 * h);
        double inv_2h3 = 1.0 / (2.0 * h * h * h);

        // u_t ~ (u(t+h) - u(t-h)) / 2h
        auto u_t = (u_t_p - u_t_m) * inv_2h;

        // u_x ~ (u(x+h) - u(x-h)) / 2h
        auto u_x = (u_x_p - u_x_m) * inv_2h;

        // u_xxx ~ (u(x+2h) - 2u(x+h) + 2u(x-h) - u(x-2h)) / (2h^3)
        auto u_xxx = (u_x_pp - u_x_p * 2.0 + u_x_m * 2.0 - u_x_mm) * inv_2h3;

        // Residual
        auto R = u_t + u_c * u_x * 6.0 + u_xxx;

        auto loss = R.pow(2.0).mean_all();

        if (iter % 100 == 0) {
            std::cout << "Iter " << iter << " Loss: " << loss.item<double>() << std::endl;
        }

        // ============================================================
        // Compute gradient weights for each stencil point (chain rule)
        // L = mean(R^2), dL/dR = 2R/N
        // ============================================================

        double N = static_cast<double>(batch_size);
        auto dL_dR = R * (2.0 / N);

        // R = u_t + 6*u_c*u_x + u_xxx
        // dR/du_tp = 1/2h, dR/du_tm = -1/2h (constant)
        // dR/du_c = 6*u_x (depends on u_x)
        // dR/du_xp = 6*u_c * 1/2h - 2/2h^3 (depends on u_c)
        // dR/du_xm = 6*u_c * (-1/2h) + 2/2h^3 (depends on u_c)
        // dR/du_xpp = 1/2h^3, dR/du_xmm = -1/2h^3 (constant)

        double w_tp = inv_2h;
        double w_tm = -inv_2h;
        double w_xpp = inv_2h3;
        double w_xmm = -inv_2h3;

        // For weights that depend on other outputs, we need to compute them separately
        // dR/du_c = 6*u_x
        auto grad_u_c = dL_dR * u_x * 6.0;

        // dR/du_xp = 6*u_c * 1/2h - 2/2h^3
        auto grad_u_xp = dL_dR * (u_c * 6.0 * inv_2h - 2.0 * inv_2h3);

        // dR/du_xm = -6*u_c * 1/2h + 2/2h^3
        auto grad_u_xm = dL_dR * (u_c * (-6.0) * inv_2h + 2.0 * inv_2h3);

        // dR/du_tp = 1/2h, dR/du_tm = -1/2h
        auto grad_u_tp = dL_dR * inv_2h;
        auto grad_u_tm = dL_dR * (-inv_2h);

        // dR/du_xpp = 1/2h^3, dR/du_xmm = -1/2h^3
        auto grad_u_xpp = dL_dR * inv_2h3;
        auto grad_u_xmm = dL_dR * (-inv_2h3);

        // ============================================================
        // Backward pass through all stencil points using for loop
        // Note: For center point, we need to forward again since forward_all cached last
        // ============================================================

        // For points with tensor-dependent gradients, we need to re-forward
        net.forward(stencils[0].input); net.backward(grad_u_c);
        net.forward(stencils[1].input); net.backward(grad_u_xp);
        net.forward(stencils[2].input); net.backward(grad_u_xm);
        net.forward(stencils[3].input); net.backward(grad_u_xpp);
        net.forward(stencils[4].input); net.backward(grad_u_xmm);
        net.forward(stencils[5].input); net.backward(grad_u_tp);
        net.forward(stencils[6].input); net.backward(grad_u_tm);

        optimizer.step();
    }

    std::cout << "Training finished." << std::endl;
    return 0;
}
