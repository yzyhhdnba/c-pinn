#!/usr/bin/env python3
"""Extract summary statistics from torch_results.json"""
import json
import os

base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base, "torch_results.json")) as f:
    data = json.load(f)

print("=" * 70)
print("TEST 1: Statistical summary (5 runs, 1000 iters, default config)")
print("=" * 70)
for eq in ["kdv", "sine_gordon", "allen_cahn"]:
    runs = data[f"stats_{eq}"]
    times = [r["total_time_s"] for r in runs]
    losses = [r["final_loss"] for r in runs]
    mem = [r["rss_peak_mb"] for r in runs]
    n = len(times)
    mean_t = sum(times) / n
    std_t = (sum((x - mean_t) ** 2 for x in times) / (n - 1)) ** 0.5
    mean_l = sum(losses) / n
    min_l = min(losses)
    max_l = max(losses)
    median_l = sorted(losses)[n // 2]
    print(f"\n{eq}:")
    print(f"  time:  {mean_t:.3f} ± {std_t:.3f}s  (individual: {', '.join(f'{t:.3f}' for t in times)})")
    print(f"  loss:  mean={mean_l:.3e}  median={median_l:.3e}  min={min_l:.3e}  max={max_l:.3e}")
    print(f"  RSS:   {mem[0]:.1f} MB")
    print(f"  params: {runs[0]['n_params']}")

print("\n")
print("=" * 70)
print("TEST 1: Per-iteration timing breakdown (run 1)")
print("=" * 70)
for eq in ["kdv", "sine_gordon", "allen_cahn"]:
    iters = data[f"stats_{eq}"][0]["iter_times"]
    times_ms = [x["time_ms"] for x in iters]
    n = len(times_ms)
    first10 = sum(times_ms[:10]) / 10
    mid10 = sum(times_ms[n // 2 : n // 2 + 10]) / 10
    last10 = sum(times_ms[-10:]) / 10
    mean_all = sum(times_ms) / n
    max_t = max(times_ms)
    min_t = min(times_ms)
    print(f"  {eq}: avg={mean_all:.2f}ms  first10={first10:.2f}ms  mid={mid10:.2f}ms  last10={last10:.2f}ms  min={min_t:.2f}ms  max={max_t:.2f}ms")

print("\n")
print("=" * 70)
print("TEST 2: Batch size scaling")
print("=" * 70)
for eq in ["kdv", "sine_gordon", "allen_cahn"]:
    print(f"\n{eq}:")
    for r in data[f"batch_{eq}"]:
        print(f"  batch={r['batch_size']:>4d}  time={r['total_time_s']:.3f}s  loss={r['final_loss']:.3e}")

print("\n")
print("=" * 70)
print("TEST 3: Network width scaling")
print("=" * 70)
for eq in ["kdv", "sine_gordon", "allen_cahn"]:
    print(f"\n{eq}:")
    for r in data[f"width_{eq}"]:
        w = r["layers"][1]
        print(f"  width={w:>4d}  params={r['n_params']:>6d}  time={r['total_time_s']:.3f}s  loss={r['final_loss']:.3e}")

print("\n")
print("=" * 70)
print("TEST 4: Long training (5000 iterations)")
print("=" * 70)
for eq in ["kdv", "sine_gordon", "allen_cahn"]:
    r = data[f"long_{eq}"]
    # Show convergence milestones
    losses = r["losses"]
    milestones = {}
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    for entry in losses:
        for th in thresholds:
            if th not in milestones and entry["loss"] < th:
                milestones[th] = entry["iter"]
    
    print(f"\n{eq}:  total_time={r['total_time_s']:.3f}s  final_loss={r['final_loss']:.3e}")
    print(f"  Convergence milestones: ", end="")
    for th in thresholds:
        if th in milestones:
            print(f"<{th:.0e} @iter {milestones[th]}", end="  ")
        else:
            print(f"<{th:.0e}: not reached", end="  ")
    print()
    # Show loss at key iterations
    key_iters = [0, 100, 500, 1000, 2000, 3000, 4000, 4990]
    print("  Loss trajectory: ", end="")
    for entry in losses:
        if entry["iter"] in key_iters:
            print(f"iter{entry['iter']}={entry['loss']:.2e}", end="  ")
    print()
