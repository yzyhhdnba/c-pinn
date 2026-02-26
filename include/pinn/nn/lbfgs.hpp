#pragma once

#include "pinn/nn/gemm.hpp"  // TYPE_VAL

namespace pinn::nn {

struct LbfgsParams {
    int max_iters{200};
    int history_size{10};
    TYPE_VAL grad_tol{1e-8};

    // Backtracking (Armijo)
    int max_linesearch{20};
    TYPE_VAL c1{1e-4};
    TYPE_VAL step_init{1.0};
    TYPE_VAL step_min{1e-20};
    TYPE_VAL step_shrink{0.5};

    // Curvature safeguard
    TYPE_VAL curvature_eps{1e-12};
};

// f(x) and gradient g(x) callback: returns f, writes g (size n)
using LbfgsEvalFn = TYPE_VAL (*)(void* ctx, const TYPE_VAL* x, TYPE_VAL* g, int n);

struct LbfgsState {
    int n{0};
    int m{0};
    int k{0};
    int head{0};     // next insert slot
    int size{0};     // filled slots [0..m]

    // history: s and y are [m * n]
    TYPE_VAL* s{nullptr};
    TYPE_VAL* y{nullptr};
    TYPE_VAL* rho{nullptr};

    // workspace
    TYPE_VAL* alpha{nullptr};
    TYPE_VAL* q{nullptr};
    TYPE_VAL* r{nullptr};
    TYPE_VAL* d{nullptr};
    TYPE_VAL* x_prev{nullptr};
    TYPE_VAL* g_prev{nullptr};
    TYPE_VAL* x_trial{nullptr};
    TYPE_VAL* g_trial{nullptr};
};

int lbfgs_init(LbfgsState* st, int n, int history_size);
void lbfgs_free(LbfgsState* st);

// Optimize in-place x (size n). Returns 0 on success.
int lbfgs_optimize(LbfgsState* st,
                   TYPE_VAL* x,
                   int n,
                   LbfgsEvalFn eval,
                   void* ctx,
                   const LbfgsParams& params,
                   TYPE_VAL* out_fx);

}  // namespace pinn::nn
