#include "pinn/core/tensor.hpp"
#include "pinn/core/rng.hpp"
#include "pinn/nn/fnn.hpp"
#include "pinn/nn/relu.hpp"
#include "pinn/nn/tanh.hpp"
#include "pinn/nn/sin.hpp"
#include "pinn/nn/adam.hpp"
#include "pinn/nn/flatten.hpp"
#include "pinn/nn/lbfgs.hpp"
#include "pinn/nn/initialization.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

using pinn::core::DType;
using pinn::core::Tensor;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAILED: " << msg << std::endl;
        std::exit(1);
    }
}

int main() {
    {
        Tensor a = Tensor::zeros({2, 3});
        check(a.defined(), "zeros defined");
        check(a.numel() == 6, "zeros numel");
        check(a.at(0, 0) == 0.0, "zeros value");

        Tensor b = Tensor::ones({2, 3});
        Tensor c = a + b;
        check(c.at(1, 2) == 1.0, "add element");

        Tensor d = b * b;
        check(d.at(0, 1) == 1.0, "mul element");

        Tensor e = Tensor::full({2, 3}, -2.0).abs();
        check(e.at(1, 1) == 2.0, "abs");

        Tensor f = Tensor::full({2, 3}, 9.0).sqrt();
        check(f.at(0, 2) == 3.0, "sqrt");

        Tensor g = Tensor::full({2, 3}, -1.0).relu();
        check(g.at(1, 2) == 0.0, "relu");
    }

    {
        Tensor x = Tensor::linspace(0.0, 1.0, 5);
        check(x.shape() == Tensor::Shape({5}), "linspace shape");
        check(std::abs(x.at(4) - 1.0) < 1e-12, "linspace last");

        auto top = x.topk(2);
        check(top.first.dtype() == DType::kFloat64, "topk values dtype");
        check(top.second.dtype() == DType::kInt64, "topk idx dtype");
        check(std::abs(top.first.at(0) - 1.0) < 1e-12, "topk max");
        check(top.second.at_int(0) == 4, "topk max idx");
    }

    {
        Tensor a = Tensor::full({4, 2}, 0.0);
        for (int64_t i = 0; i < 4; ++i) {
            a.set(i, 0, static_cast<double>(i));
            a.set(i, 1, static_cast<double>(i + 10));
        }
        Tensor b = a.slice(0, 1, 3);
        check(b.shape() == Tensor::Shape({2, 2}), "slice shape");
        check(b.at(0, 0) == 1.0 && b.at(1, 1) == 12.0, "slice values");

        Tensor t = b.transpose(0, 1);
        check(t.shape() == Tensor::Shape({2, 2}), "transpose shape");
        check(t.at(0, 1) == 2.0 && t.at(1, 0) == 11.0, "transpose values");

        Tensor r = t.reshape({4});
        check(r.shape() == Tensor::Shape({4}), "reshape shape");
        check(r.numel() == 4, "reshape numel");

        Tensor c = Tensor::cat({b, b}, 0);
        check(c.shape() == Tensor::Shape({4, 2}), "cat shape");
        check(c.at(2, 1) == 11.0, "cat value");

        Tensor s = Tensor::stack({b, b}, 0);
        check(s.shape() == Tensor::Shape({2, 2, 2}), "stack shape");
    }

    {
        Tensor p = Tensor::full({3}, 2.0);
        Tensor q = Tensor::full({3}, 4.0);
        Tensor loss = pinn::core::mse_loss(p, q);
        check(std::abs(loss.item<double>() - 4.0) < 1e-12, "mse_loss");
    }

    {
        pinn::core::Rng rng(123);
        pinn::nn::Linear lin(2, 3, pinn::nn::InitType::kZero, 0.0, rng);

        Tensor x({2, 2});
        x.set(0, 0, 1.0);
        x.set(0, 1, 2.0);
        x.set(1, 0, 3.0);
        x.set(1, 1, 4.0);

        Tensor y = lin.forward(x);
        check(y.shape() == Tensor::Shape({2, 3}), "linear shape");
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                check(std::abs(y.at(i, j)) < 1e-12, "linear zeros");
            }
        }
    }

    {
        // C-style ReLU forward/backward (in-place)
        Tensor feats({2, 3});
        double* f = feats.data_ptr<double>();
        f[0] = -1.0;
        f[1] = 0.0;
        f[2] = 2.0;
        f[3] = -3.0;
        f[4] = 4.0;
        f[5] = 5.0;
        pinn::nn::relu(reinterpret_cast<pinn::nn::TYPE_VAL*>(f), 2, 3);
        check(feats.at(0, 0) == 0.0 && feats.at(0, 2) == 2.0, "relu forward");
        check(feats.at(1, 0) == 0.0 && feats.at(1, 1) == 4.0, "relu forward 2");

        Tensor delta = Tensor::ones({2, 3});
        pinn::nn::relu_backward(reinterpret_cast<pinn::nn::TYPE_VAL*>(delta.data_ptr<double>()),
                                reinterpret_cast<const pinn::nn::TYPE_VAL*>(feats.data_ptr<double>()),
                                2,
                                3);
        // entries where feats==0 should be masked to 0
        check(delta.at(0, 0) == 0.0 && delta.at(0, 1) == 0.0, "relu backward mask");
        check(delta.at(0, 2) == 1.0 && delta.at(1, 1) == 1.0, "relu backward pass");
    }

    {
        // C-style tanh forward/backward (in-place)
        Tensor feats({2, 3});
        double* f = feats.data_ptr<double>();
        f[0] = -2.0;
        f[1] = -0.5;
        f[2] = 0.0;
        f[3] = 0.5;
        f[4] = 2.0;
        f[5] = 3.0;

        pinn::nn::tanh(reinterpret_cast<pinn::nn::TYPE_VAL*>(f), 2, 3);
        // Validate numerically against std::tanh on original inputs
        const double x0[6] = {-2.0, -0.5, 0.0, 0.5, 2.0, 3.0};
        for (int i = 0; i < 6; ++i) {
            check(std::abs(f[i] - std::tanh(x0[i])) < 1e-12, "tanh forward");
        }

        Tensor delta = Tensor::ones({2, 3});
        pinn::nn::tanh_backward(reinterpret_cast<pinn::nn::TYPE_VAL*>(delta.data_ptr<double>()),
                                reinterpret_cast<const pinn::nn::TYPE_VAL*>(feats.data_ptr<double>()),
                                2,
                                3);
        for (int i = 0; i < 6; ++i) {
            const double y = f[i];
            const double expected = 1.0 - y * y;
            check(std::abs(delta.data_ptr<double>()[i] - expected) < 1e-12, "tanh backward");
        }
    }

    {
        // C-style sin forward/backward (in-place)
        Tensor x({2, 3});
        double* xv = x.data_ptr<double>();
        xv[0] = -2.0;
        xv[1] = -0.5;
        xv[2] = 0.0;
        xv[3] = 0.5;
        xv[4] = 2.0;
        xv[5] = 3.0;

        Tensor x_in = x.clone();
        pinn::nn::sin(reinterpret_cast<pinn::nn::TYPE_VAL*>(xv), 2, 3);
        for (int i = 0; i < 6; ++i) {
            check(std::abs(xv[i] - std::sin(x_in.data_ptr<double>()[i])) < 1e-12, "sin forward");
        }

        Tensor delta = Tensor::ones({2, 3});
        pinn::nn::sin_backward(reinterpret_cast<pinn::nn::TYPE_VAL*>(delta.data_ptr<double>()),
                               reinterpret_cast<const pinn::nn::TYPE_VAL*>(x_in.data_ptr<double>()),
                               2,
                               3);
        for (int i = 0; i < 6; ++i) {
            const double expected = std::cos(x_in.data_ptr<double>()[i]);
            check(std::abs(delta.data_ptr<double>()[i] - expected) < 1e-12, "sin backward");
        }
    }

    {
        // Adam update sanity (t=1, beta1=0.9, beta2=0.999)
        // With m=v=0 initially and weight_decay=0:
        // m = (1-b1)*g, v=(1-b2)*g^2, m_hat=g, v_hat=g^2 => update = lr*sign(g)/(1+eps/|g|)
        const int layer_dims[2] = {1, 1};
        pinn::nn::TYPE_VAL* m_w = nullptr;
        pinn::nn::TYPE_VAL* v_w = nullptr;
        pinn::nn::TYPE_VAL* m_b = nullptr;
        pinn::nn::TYPE_VAL* v_b = nullptr;
        check(pinn::nn::init_adam(&m_w, &v_w, &m_b, &v_b, 1, layer_dims) == 0, "init_adam");

        pinn::nn::TYPE_VAL w[1] = {1.0};
        pinn::nn::TYPE_VAL b[1] = {0.0};
        pinn::nn::TYPE_VAL gw[1] = {2.0};
        pinn::nn::TYPE_VAL gb[1] = {-3.0};

        const float lr = 0.1f;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        check(pinn::nn::gcn_update_adam(w, b, gw, gb,
                                       m_w, v_w, m_b, v_b,
                                       1, layer_dims,
                                       lr, 1,
                                       beta1, beta2, eps,
                                       0.0f) == 0,
              "adam update");

        // Expected approx: w -= 0.1 * 1 = 0.9, b -= 0.1 * (-1) = +0.1
        check(std::abs(w[0] - 0.9) < 1e-6, "adam weight");
        check(std::abs(b[0] - 0.1) < 1e-6, "adam bias");

        std::free(m_w);
        std::free(v_w);
        std::free(m_b);
        std::free(v_b);
    }

    {
        // Flat mapping: pack -> modify -> unpack should affect forward.
        pinn::nn::Fnn net({2, 2, 1}, "relu", pinn::nn::InitType::kZero, 0.0, 42);
        const auto layout = pinn::nn::make_flat_layout(net.layer_sizes());

        std::vector<pinn::nn::TYPE_VAL> w(static_cast<size_t>(layout.size_w), 0.0);
        std::vector<pinn::nn::TYPE_VAL> b(static_cast<size_t>(layout.size_b), 0.0);
        check(pinn::nn::pack_params(net, w.data(), b.data(), layout) == 0, "pack_params");

        // Set first layer weights to identity, second layer weights to [1,1], biases zero.
        // Layer0: in=2,out=2 weights: [[1,0],[0,1]] (row-major)
        w[layout.w_offsets[0] + 0] = 1.0;
        w[layout.w_offsets[0] + 1] = 0.0;
        w[layout.w_offsets[0] + 2] = 0.0;
        w[layout.w_offsets[0] + 3] = 1.0;
        // Layer1: in=2,out=1 weights: [[1],[1]]
        w[layout.w_offsets[1] + 0] = 1.0;
        w[layout.w_offsets[1] + 1] = 1.0;

        check(pinn::nn::unpack_params(net, w.data(), b.data(), layout) == 0, "unpack_params");

        Tensor x({1, 2});
        x.set(0, 0, 2.0);
        x.set(0, 1, 3.0);
        Tensor y = net.forward(x);
        // y = relu([2,3]) dot [1,1] = 5
        check(std::abs(y.at(0, 0) - 5.0) < 1e-12, "flat mapping forward");
    }

    {
        // L-BFGS on a simple quadratic: f(x)=0.5*sum_i a_i x_i^2 - sum_i b_i x_i
        struct QuadCtx {
            int n;
            std::vector<double> a;
            std::vector<double> b;
        } ctx;
        ctx.n = 4;
        ctx.a = {1.0, 2.0, 3.0, 4.0};
        ctx.b = {1.0, -2.0, 3.0, -4.0};

        auto eval = [](void* p, const pinn::nn::TYPE_VAL* x, pinn::nn::TYPE_VAL* g, int n) -> pinn::nn::TYPE_VAL {
            auto* c = static_cast<QuadCtx*>(p);
            (void)n;
            double f = 0.0;
            for (int i = 0; i < c->n; ++i) {
                const double ai = c->a[static_cast<size_t>(i)];
                const double bi = c->b[static_cast<size_t>(i)];
                const double xi = x[i];
                g[i] = ai * xi - bi;
                f += 0.5 * ai * xi * xi - bi * xi;
            }
            return f;
        };

        pinn::nn::LbfgsState st;
        check(pinn::nn::lbfgs_init(&st, ctx.n, 6) == 0, "lbfgs_init");
        pinn::nn::LbfgsParams p;
        p.max_iters = 100;
        p.grad_tol = 1e-10;
        p.history_size = 6;

        pinn::nn::TYPE_VAL x[4] = {0.0, 0.0, 0.0, 0.0};
        pinn::nn::TYPE_VAL fx = 0.0;
        const int rc = pinn::nn::lbfgs_optimize(&st, x, ctx.n, eval, &ctx, p, &fx);
        check(rc == 0 || rc == 1, "lbfgs_optimize rc");

        // optimum x* = b/a
        for (int i = 0; i < ctx.n; ++i) {
            const double x_star = ctx.b[static_cast<size_t>(i)] / ctx.a[static_cast<size_t>(i)];
            check(std::abs(x[i] - x_star) < 1e-6, "lbfgs solution");
        }

        pinn::nn::lbfgs_free(&st);
    }

    {
        // C-style Xavier/Glorot uniform via rand()/srand(0): deterministic and within bounds.
        int layer_dims[3] = {2, 3, 1};
        const int num_layers = 2;

        pinn::nn::TYPE_VAL* w0a = static_cast<pinn::nn::TYPE_VAL*>(std::malloc(sizeof(pinn::nn::TYPE_VAL) * 2 * 3));
        pinn::nn::TYPE_VAL* w1a = static_cast<pinn::nn::TYPE_VAL*>(std::malloc(sizeof(pinn::nn::TYPE_VAL) * 3 * 1));
        pinn::nn::TYPE_VAL* wa[num_layers] = {w0a, w1a};
        check(pinn::nn::init_weights_xavier_uniform(wa, num_layers, layer_dims) == 0, "xavier_uniform init rc");

        pinn::nn::TYPE_VAL* w0b = static_cast<pinn::nn::TYPE_VAL*>(std::malloc(sizeof(pinn::nn::TYPE_VAL) * 2 * 3));
        pinn::nn::TYPE_VAL* w1b = static_cast<pinn::nn::TYPE_VAL*>(std::malloc(sizeof(pinn::nn::TYPE_VAL) * 3 * 1));
        pinn::nn::TYPE_VAL* wb[num_layers] = {w0b, w1b};
        check(pinn::nn::init_weights_xavier_uniform(wb, num_layers, layer_dims) == 0, "xavier_uniform init rc 2");

        // Determinism: because function calls srand(0) internally.
        for (int i = 0; i < 2 * 3; ++i) {
            check(std::abs(w0a[i] - w0b[i]) < 1e-15, "xavier_uniform deterministic layer0");
        }
        for (int i = 0; i < 3 * 1; ++i) {
            check(std::abs(w1a[i] - w1b[i]) < 1e-15, "xavier_uniform deterministic layer1");
        }

        // Range checks: each layer has its own bound a.
        const double a0 = std::sqrt(6.0 / static_cast<double>(layer_dims[0] + layer_dims[1]));
        const double a1 = std::sqrt(6.0 / static_cast<double>(layer_dims[1] + layer_dims[2]));
        for (int i = 0; i < 2 * 3; ++i) {
            check(w0a[i] >= -a0 - 1e-15 && w0a[i] <= a0 + 1e-15, "xavier_uniform range layer0");
        }
        for (int i = 0; i < 3 * 1; ++i) {
            check(w1a[i] >= -a1 - 1e-15 && w1a[i] <= a1 + 1e-15, "xavier_uniform range layer1");
        }

        std::free(w0a);
        std::free(w1a);
        std::free(w0b);
        std::free(w1b);
    }

    {
        // C-style Kaiming/He uniform via rand()/srand(0): deterministic and within bounds.
        int layer_dims[3] = {2, 3, 1};
        const int num_layers = 2;

        pinn::nn::TYPE_VAL* w0a = static_cast<pinn::nn::TYPE_VAL*>(std::malloc(sizeof(pinn::nn::TYPE_VAL) * 2 * 3));
        pinn::nn::TYPE_VAL* w1a = static_cast<pinn::nn::TYPE_VAL*>(std::malloc(sizeof(pinn::nn::TYPE_VAL) * 3 * 1));
        pinn::nn::TYPE_VAL* wa[num_layers] = {w0a, w1a};
        check(pinn::nn::init_weights_kaiming_uniform(wa, num_layers, layer_dims) == 0, "kaiming_uniform init rc");

        pinn::nn::TYPE_VAL* w0b = static_cast<pinn::nn::TYPE_VAL*>(std::malloc(sizeof(pinn::nn::TYPE_VAL) * 2 * 3));
        pinn::nn::TYPE_VAL* w1b = static_cast<pinn::nn::TYPE_VAL*>(std::malloc(sizeof(pinn::nn::TYPE_VAL) * 3 * 1));
        pinn::nn::TYPE_VAL* wb[num_layers] = {w0b, w1b};
        check(pinn::nn::init_weights_kaiming_uniform(wb, num_layers, layer_dims) == 0, "kaiming_uniform init rc 2");

        for (int i = 0; i < 2 * 3; ++i) {
            check(std::abs(w0a[i] - w0b[i]) < 1e-15, "kaiming_uniform deterministic layer0");
        }
        for (int i = 0; i < 3 * 1; ++i) {
            check(std::abs(w1a[i] - w1b[i]) < 1e-15, "kaiming_uniform deterministic layer1");
        }

        const double a0 = std::sqrt(6.0 / static_cast<double>(layer_dims[0]));
        const double a1 = std::sqrt(6.0 / static_cast<double>(layer_dims[1]));
        for (int i = 0; i < 2 * 3; ++i) {
            check(w0a[i] >= -a0 - 1e-15 && w0a[i] <= a0 + 1e-15, "kaiming_uniform range layer0");
        }
        for (int i = 0; i < 3 * 1; ++i) {
            check(w1a[i] >= -a1 - 1e-15 && w1a[i] <= a1 + 1e-15, "kaiming_uniform range layer1");
        }

        std::free(w0a);
        std::free(w1a);
        std::free(w0b);
        std::free(w1b);
    }

    std::cout << "tensor_test OK" << std::endl;
    return 0;
}
