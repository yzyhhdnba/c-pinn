#!/usr/bin/env python3
"""
Comprehensive PyTorch PINN benchmark: KdV, Sine-Gordon, Allen-Cahn.
Collects per-iteration loss, per-iteration wall time, memory usage,
supports multiple runs for statistical significance, and different
batch sizes / network widths for scalability testing.

Output: JSON files with full data for report generation.
"""
import json
import os
import sys
import time
import tracemalloc
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ================================================================
# PDE residual functions
# ================================================================

def residual_kdv(model, x_t):
    """KdV: u_t + 6*u*u_x + u_xxx = 0"""
    x_t.requires_grad_(True)
    u = model(x_t)
    grads = torch.autograd.grad(u, x_t, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_xxx = torch.autograd.grad(u_xx, x_t, torch.ones_like(u_xx), create_graph=True)[0][:, 0:1]
    return u_t + 6.0 * u * u_x + u_xxx


def residual_sine_gordon(model, x_t):
    """Sine-Gordon: u_tt - u_xx + sin(u) = 0"""
    x_t.requires_grad_(True)
    u = model(x_t)
    grads = torch.autograd.grad(u, x_t, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t, x_t, torch.ones_like(u_t), create_graph=True)[0][:, 1:2]
    return u_tt - u_xx + torch.sin(u)


def residual_allen_cahn(model, x_t):
    """Allen-Cahn: u_t - 0.0001*u_xx + 5*(u^3 - u) = 0"""
    x_t.requires_grad_(True)
    u = model(x_t)
    grads = torch.autograd.grad(u, x_t, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, x_t, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return u_t - 0.0001 * u_xx + 5.0 * (u ** 3 - u)


EQUATIONS = {
    "kdv": residual_kdv,
    "sine_gordon": residual_sine_gordon,
    "allen_cahn": residual_allen_cahn,
}


# ================================================================
# Training function with full instrumentation
# ================================================================

def train_pinn(equation_name, layers, batch_size, iterations, seed, record_every=1):
    """
    Train a PINN and return detailed per-iteration metrics.
    """
    torch.manual_seed(seed)
    model = PINN(layers).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    residual_fn = EQUATIONS[equation_name]
    n_params = model.count_parameters()

    # Memory tracking
    tracemalloc.start()
    mem_before = tracemalloc.get_traced_memory()[1]  # peak so far

    losses = []
    iter_times = []

    t_total_start = time.perf_counter()

    for it in range(iterations):
        t_iter_start = time.perf_counter()

        optimizer.zero_grad()
        x_t = torch.rand(batch_size, 2, dtype=torch.float64)
        residual = residual_fn(model, x_t)
        loss = torch.mean(residual ** 2)
        loss.backward()
        optimizer.step()

        t_iter_end = time.perf_counter()

        if it % record_every == 0:
            losses.append({"iter": it, "loss": loss.item()})
            iter_times.append({"iter": it, "time_ms": (t_iter_end - t_iter_start) * 1000})

    total_time = time.perf_counter() - t_total_start
    mem_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    mem_used_mb = (mem_peak - mem_before) / (1024 * 1024)
    # Also measure RSS via resource module
    import resource
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    return {
        "equation": equation_name,
        "layers": layers,
        "batch_size": batch_size,
        "iterations": iterations,
        "seed": seed,
        "n_params": n_params,
        "total_time_s": total_time,
        "mem_peak_tracemalloc_mb": round(mem_used_mb, 2),
        "rss_peak_mb": round(rss_mb, 2),
        "final_loss": losses[-1]["loss"],
        "losses": losses,
        "iter_times": iter_times,
    }


def run_benchmark_suite():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    equations = ["kdv", "sine_gordon", "allen_cahn"]
    default_layers = [2, 50, 50, 50, 1]

    # ────────────────────────────────────────────
    # Test 1: Default config, 5 runs each for stats
    # ────────────────────────────────────────────
    print("=" * 60)
    print("TEST 1: Default config × 5 runs (statistical)")
    print("=" * 60)
    for eq in equations:
        runs = []
        for run_i in range(5):
            seed = 42 + run_i
            print(f"  {eq} run {run_i+1}/5 (seed={seed}) ...", end=" ", flush=True)
            r = train_pinn(eq, default_layers, batch_size=64, iterations=1000,
                           seed=seed, record_every=10)
            print(f"loss={r['final_loss']:.6e}  time={r['total_time_s']:.3f}s")
            runs.append(r)
        results[f"stats_{eq}"] = runs

    # ────────────────────────────────────────────
    # Test 2: Batch size scaling (32, 64, 128, 256, 512)
    # ────────────────────────────────────────────
    print("=" * 60)
    print("TEST 2: Batch size scaling")
    print("=" * 60)
    for eq in equations:
        batch_results = []
        for bs in [32, 64, 128, 256, 512]:
            print(f"  {eq} batch={bs} ...", end=" ", flush=True)
            r = train_pinn(eq, default_layers, batch_size=bs, iterations=1000,
                           seed=42, record_every=50)
            print(f"loss={r['final_loss']:.6e}  time={r['total_time_s']:.3f}s")
            batch_results.append(r)
        results[f"batch_{eq}"] = batch_results

    # ────────────────────────────────────────────
    # Test 3: Network width scaling
    # [2,20,20,20,1] → [2,50,50,50,1] → [2,100,100,100,1] → [2,200,200,200,1]
    # ────────────────────────────────────────────
    print("=" * 60)
    print("TEST 3: Network width scaling")
    print("=" * 60)
    widths = [20, 50, 100, 200]
    for eq in equations:
        width_results = []
        for w in widths:
            layers_w = [2, w, w, w, 1]
            print(f"  {eq} width={w} ({layers_w}) ...", end=" ", flush=True)
            r = train_pinn(eq, layers_w, batch_size=64, iterations=1000,
                           seed=42, record_every=50)
            print(f"loss={r['final_loss']:.6e}  time={r['total_time_s']:.3f}s  params={r['n_params']}")
            width_results.append(r)
        results[f"width_{eq}"] = width_results

    # ────────────────────────────────────────────
    # Test 4: Long training (5000 iterations) for deeper convergence
    # ────────────────────────────────────────────
    print("=" * 60)
    print("TEST 4: Long training (5000 iterations)")
    print("=" * 60)
    for eq in equations:
        print(f"  {eq} 5000 iters ...", end=" ", flush=True)
        r = train_pinn(eq, default_layers, batch_size=64, iterations=5000,
                       seed=42, record_every=10)
        print(f"loss={r['final_loss']:.6e}  time={r['total_time_s']:.3f}s")
        results[f"long_{eq}"] = r

    # Save all results
    out_path = os.path.join(out_dir, "torch_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    run_benchmark_suite()
