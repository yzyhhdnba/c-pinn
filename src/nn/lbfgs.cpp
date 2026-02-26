#include "pinn/nn/lbfgs.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace pinn::nn {

namespace {

TYPE_VAL dot(const TYPE_VAL* a, const TYPE_VAL* b, int n) {
    TYPE_VAL s = 0;
    for (int i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

TYPE_VAL norm2(const TYPE_VAL* a, int n) {
    return std::sqrt(dot(a, a, n));
}

void axpy(TYPE_VAL* y, const TYPE_VAL* x, TYPE_VAL a, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

void copy(TYPE_VAL* dst, const TYPE_VAL* src, int n) {
    std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(TYPE_VAL));
}

void scal(TYPE_VAL* x, TYPE_VAL a, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] *= a;
    }
}

int slot_from_newest(const LbfgsState* st, int newest_index) {
    // newest_index=0 => most recent
    int idx = st->head - 1 - newest_index;
    while (idx < 0) {
        idx += st->m;
    }
    return idx % st->m;
}

TYPE_VAL* s_ptr(LbfgsState* st, int slot) {
    return st->s + static_cast<size_t>(slot) * static_cast<size_t>(st->n);
}

TYPE_VAL* y_ptr(LbfgsState* st, int slot) {
    return st->y + static_cast<size_t>(slot) * static_cast<size_t>(st->n);
}

const TYPE_VAL* s_ptr_c(const LbfgsState* st, int slot) {
    return st->s + static_cast<size_t>(slot) * static_cast<size_t>(st->n);
}

const TYPE_VAL* y_ptr_c(const LbfgsState* st, int slot) {
    return st->y + static_cast<size_t>(slot) * static_cast<size_t>(st->n);
}

}  // namespace

int lbfgs_init(LbfgsState* st, int n, int history_size) {
    if (!st || n <= 0 || history_size <= 0) {
        return -1;
    }
    std::memset(st, 0, sizeof(LbfgsState));
    st->n = n;
    st->m = history_size;

    const size_t mn = static_cast<size_t>(history_size) * static_cast<size_t>(n);
    st->s = static_cast<TYPE_VAL*>(std::malloc(mn * sizeof(TYPE_VAL)));
    st->y = static_cast<TYPE_VAL*>(std::malloc(mn * sizeof(TYPE_VAL)));
    st->rho = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(history_size) * sizeof(TYPE_VAL)));

    st->alpha = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(history_size) * sizeof(TYPE_VAL)));
    st->q = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(n) * sizeof(TYPE_VAL)));
    st->r = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(n) * sizeof(TYPE_VAL)));
    st->d = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(n) * sizeof(TYPE_VAL)));
    st->x_prev = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(n) * sizeof(TYPE_VAL)));
    st->g_prev = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(n) * sizeof(TYPE_VAL)));
    st->x_trial = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(n) * sizeof(TYPE_VAL)));
    st->g_trial = static_cast<TYPE_VAL*>(std::malloc(static_cast<size_t>(n) * sizeof(TYPE_VAL)));

    if (!st->s || !st->y || !st->rho || !st->alpha || !st->q || !st->r || !st->d || !st->x_prev || !st->g_prev ||
        !st->x_trial || !st->g_trial) {
        lbfgs_free(st);
        return -2;
    }

    std::memset(st->s, 0, mn * sizeof(TYPE_VAL));
    std::memset(st->y, 0, mn * sizeof(TYPE_VAL));
    std::memset(st->rho, 0, static_cast<size_t>(history_size) * sizeof(TYPE_VAL));
    return 0;
}

void lbfgs_free(LbfgsState* st) {
    if (!st) {
        return;
    }
    std::free(st->s);
    std::free(st->y);
    std::free(st->rho);
    std::free(st->alpha);
    std::free(st->q);
    std::free(st->r);
    std::free(st->d);
    std::free(st->x_prev);
    std::free(st->g_prev);
    std::free(st->x_trial);
    std::free(st->g_trial);
    std::memset(st, 0, sizeof(LbfgsState));
}

static void two_loop_recursion(LbfgsState* st, const TYPE_VAL* g, const LbfgsParams& params) {
    // q = g
    copy(st->q, g, st->n);

    // First loop: newest -> oldest
    for (int i = 0; i < st->size; ++i) {
        const int slot = slot_from_newest(st, i);
        const TYPE_VAL* s = s_ptr_c(st, slot);
        const TYPE_VAL* y = y_ptr_c(st, slot);
        const TYPE_VAL rho = st->rho[slot];
        const TYPE_VAL a = rho * dot(s, st->q, st->n);
        st->alpha[i] = a;
        // q = q - a*y
        axpy(st->q, y, -a, st->n);
    }

    // Scaling of initial Hessian: gamma = (s_last^T y_last) / (y_last^T y_last)
    TYPE_VAL gamma = 1.0;
    if (st->size > 0) {
        const int slot_last = slot_from_newest(st, 0);
        const TYPE_VAL* s_last = s_ptr_c(st, slot_last);
        const TYPE_VAL* y_last = y_ptr_c(st, slot_last);
        const TYPE_VAL sy = dot(s_last, y_last, st->n);
        const TYPE_VAL yy = dot(y_last, y_last, st->n);
        if (yy > params.curvature_eps) {
            gamma = sy / yy;
        }
    }

    copy(st->r, st->q, st->n);
    scal(st->r, gamma, st->n);

    // Second loop: oldest -> newest
    for (int i = st->size - 1; i >= 0; --i) {
        const int newest_i = i;  // alpha stored by newest order
        const int slot = slot_from_newest(st, newest_i);
        const TYPE_VAL* s = s_ptr_c(st, slot);
        const TYPE_VAL* y = y_ptr_c(st, slot);
        const TYPE_VAL rho = st->rho[slot];
        const TYPE_VAL beta = rho * dot(y, st->r, st->n);
        const TYPE_VAL a = st->alpha[newest_i];
        // r = r + s*(a - beta)
        axpy(st->r, s, (a - beta), st->n);
    }

    // d = -r
    copy(st->d, st->r, st->n);
    scal(st->d, -1.0, st->n);
}

static int line_search_armijo(LbfgsState* st,
                              TYPE_VAL* x,
                              const TYPE_VAL* g,
                              TYPE_VAL fx,
                              LbfgsEvalFn eval,
                              void* ctx,
                              const LbfgsParams& params,
                              TYPE_VAL* out_fx,
                              TYPE_VAL* out_step) {
    TYPE_VAL step = params.step_init;
    const TYPE_VAL gtd = dot(g, st->d, st->n);
    if (gtd >= 0) {
        // Not a descent direction; fall back to steepest descent
        copy(st->d, g, st->n);
        scal(st->d, -1.0, st->n);
    }

    for (int ls = 0; ls < params.max_linesearch; ++ls) {
        // x_trial = x + step*d
        copy(st->x_trial, x, st->n);
        axpy(st->x_trial, st->d, step, st->n);

        const TYPE_VAL f_new = eval(ctx, st->x_trial, st->g_trial, st->n);
        // Armijo: f_new <= f + c1*step*g^T d
        if (f_new <= fx + params.c1 * step * gtd) {
            // accept
            copy(x, st->x_trial, st->n);
            if (out_fx) {
                *out_fx = f_new;
            }
            if (out_step) {
                *out_step = step;
            }
            // g updated by caller from st->g_trial
            return 0;
        }

        step *= params.step_shrink;
        if (step < params.step_min) {
            break;
        }
    }

    return -1;
}

int lbfgs_optimize(LbfgsState* st,
                   TYPE_VAL* x,
                   int n,
                   LbfgsEvalFn eval,
                   void* ctx,
                   const LbfgsParams& params,
                   TYPE_VAL* out_fx) {
    if (!st || !x || !eval || n <= 0 || st->n != n) {
        return -1;
    }

    // Initial eval
    TYPE_VAL fx = eval(ctx, x, st->g_prev, n);
    if (out_fx) {
        *out_fx = fx;
    }

    for (int iter = 1; iter <= params.max_iters; ++iter) {
        const TYPE_VAL gnorm = norm2(st->g_prev, n);
        if (gnorm < params.grad_tol) {
            return 0;
        }

        // Compute direction
        if (st->size == 0) {
            copy(st->d, st->g_prev, n);
            scal(st->d, -1.0, n);
        } else {
            two_loop_recursion(st, st->g_prev, params);
        }

        // Save previous x
        copy(st->x_prev, x, n);

        // Line search (updates x and st->g_trial)
        TYPE_VAL step = params.step_init;
        TYPE_VAL fx_new = fx;
        int ls_rc = line_search_armijo(st, x, st->g_prev, fx, eval, ctx, params, &fx_new, &step);
        if (ls_rc != 0) {
            // fallback: small steepest descent step
            copy(x, st->x_prev, n);
            axpy(x, st->d, params.step_min, n);
            fx_new = eval(ctx, x, st->g_trial, n);
        }

        // Compute s = x - x_prev, y = g_new - g_prev
        TYPE_VAL* s = s_ptr(st, st->head);
        TYPE_VAL* y = y_ptr(st, st->head);
        for (int i = 0; i < n; ++i) {
            s[i] = x[i] - st->x_prev[i];
            y[i] = st->g_trial[i] - st->g_prev[i];
        }

        const TYPE_VAL ys = dot(y, s, n);
        if (ys > params.curvature_eps) {
            st->rho[st->head] = 1.0 / ys;
            st->head = (st->head + 1) % st->m;
            if (st->size < st->m) {
                st->size += 1;
            }
        }

        // Move g_prev <- g_new
        copy(st->g_prev, st->g_trial, n);
        fx = fx_new;
        if (out_fx) {
            *out_fx = fx;
        }
    }

    return 1;  // max iters reached
}

}  // namespace pinn::nn
