#!/usr/bin/env python3
"""Per-stage profiling for PyTorch PINN training loops.

Outputs one machine-readable line per equation:
PROFILE_BREAKDOWN_TORCH key=value ...
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            mods.append(linear)
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def residual_kdv_profile(model, x_t):
    x_t.requires_grad_(True)

    t0 = time.perf_counter()
    u = model(x_t)
    t_forward_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    grads = torch.autograd.grad(u, x_t, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_xxx = torch.autograd.grad(u_xx, x_t, torch.ones_like(u_xx), create_graph=True)[0][:, 0:1]
    t_deriv_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    residual = u_t + 6.0 * u * u_x + u_xxx
    t_residual_math_ms = (time.perf_counter() - t2) * 1000.0

    return residual, {
        "model_forward_ms": t_forward_ms,
        "derivative_graph_ms": t_deriv_ms,
        "residual_math_ms": t_residual_math_ms,
    }


def residual_sine_gordon_profile(model, x_t):
    x_t.requires_grad_(True)

    t0 = time.perf_counter()
    u = model(x_t)
    t_forward_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    grads = torch.autograd.grad(u, x_t, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t, x_t, torch.ones_like(u_t), create_graph=True)[0][:, 1:2]
    t_deriv_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    residual = u_tt - u_xx + torch.sin(u)
    t_residual_math_ms = (time.perf_counter() - t2) * 1000.0

    return residual, {
        "model_forward_ms": t_forward_ms,
        "derivative_graph_ms": t_deriv_ms,
        "residual_math_ms": t_residual_math_ms,
    }


def residual_allen_cahn_profile(model, x_t):
    x_t.requires_grad_(True)

    t0 = time.perf_counter()
    u = model(x_t)
    t_forward_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    grads = torch.autograd.grad(u, x_t, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    t_deriv_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    residual = u_t - 0.0001 * u_xx + 5.0 * (u ** 3 - u)
    t_residual_math_ms = (time.perf_counter() - t2) * 1000.0

    return residual, {
        "model_forward_ms": t_forward_ms,
        "derivative_graph_ms": t_deriv_ms,
        "residual_math_ms": t_residual_math_ms,
    }


RESIDUAL_FNS = {
    "kdv": residual_kdv_profile,
    "sine_gordon": residual_sine_gordon_profile,
    "allen_cahn": residual_allen_cahn_profile,
}


def train_profile(equation, iterations=1000, batch_size=64, seed=42):
    torch.manual_seed(seed)
    model = PINN([2, 50, 50, 50, 1]).double()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    acc = {
        "data_gen_ms": 0.0,
        "model_forward_ms": 0.0,
        "derivative_graph_ms": 0.0,
        "residual_math_ms": 0.0,
        "mse_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_ms": 0.0,
        "iter_total_ms": 0.0,
    }

    residual_fn = RESIDUAL_FNS[equation]

    for _ in range(iterations):
        t_iter = time.perf_counter()

        opt.zero_grad()

        t0 = time.perf_counter()
        x_t = torch.rand(batch_size, 2, dtype=torch.float64)
        acc["data_gen_ms"] += (time.perf_counter() - t0) * 1000.0

        residual, stage = residual_fn(model, x_t)
        acc["model_forward_ms"] += stage["model_forward_ms"]
        acc["derivative_graph_ms"] += stage["derivative_graph_ms"]
        acc["residual_math_ms"] += stage["residual_math_ms"]

        t1 = time.perf_counter()
        loss = torch.mean(residual ** 2)
        acc["mse_ms"] += (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        loss.backward()
        acc["backward_ms"] += (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        opt.step()
        acc["optimizer_ms"] += (time.perf_counter() - t3) * 1000.0

        acc["iter_total_ms"] += (time.perf_counter() - t_iter) * 1000.0

    out = {
        "equation": equation,
        "iterations": iterations,
        "batch_size": batch_size,
        "seed": seed,
    }
    out.update(acc)

    for k in list(acc.keys()):
        out[f"avg_{k}"] = acc[k] / iterations

    return out


def to_line(result):
    fields = [
        "equation",
        "iterations",
        "batch_size",
        "avg_data_gen_ms",
        "avg_model_forward_ms",
        "avg_derivative_graph_ms",
        "avg_residual_math_ms",
        "avg_mse_ms",
        "avg_backward_ms",
        "avg_optimizer_ms",
        "avg_iter_total_ms",
    ]
    items = [f"{k}={result[k]}" for k in fields]
    return "PROFILE_BREAKDOWN_TORCH " + " ".join(items)


def main():
    results = {}
    for eq in ["kdv", "sine_gordon", "allen_cahn"]:
        r = train_profile(eq, iterations=1000, batch_size=64, seed=42)
        results[eq] = r
        print(to_line(r), flush=True)

    out_path = Path(__file__).resolve().parent / "torch_breakdown_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
