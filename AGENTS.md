# AGENTS.md

## Scope

This file captures key implementation and validation notes for the minimal C++ autodiff prototypes added to this repository.

## Project Context

- Repository: C-PINN (C++17 PINN framework with dual mode)
- Default development mode in this workspace: pure C++ (`PINN_USE_TORCH=OFF`)
- Prototype goals:
  - simulate PyTorch-style autodiff with nested graph capability (`create_graph=true` behavior)
  - extend to matrix-style forward flow (including `matmul`) for mini PINN-like training tests

## New Artifacts

- Scalar nested-graph implementation: `examples/autodiff/nested_graph.cpp`
- Scalar nested-graph CMake target: `example_autodiff_nested_graph`
- Matrix-style implementation with `matmul`: `examples/autodiff/matrix_graph.cpp`
- Matrix-style CMake target: `example_autodiff_matrix_graph`
- Technical report: `docs/autodiff/nested_graph_tech_report.md`
- Scalar detailed test script: `benchmark/autodiff/test_nested_graph.py`
- Matrix + legacy regression script: `benchmark/autodiff/test_matrix_graph.py`
- Minimal training closure script: `benchmark/autodiff/test_minimal_training.py`
- Detailed test outputs:
  - `benchmark/autodiff/results/nested_graph_test_results.json`
  - `benchmark/autodiff/results/nested_graph_test_report.md`
  - `benchmark/autodiff/results/matrix_graph_test_results.json`
  - `benchmark/autodiff/results/matrix_graph_test_report.md`
  - `benchmark/autodiff/results/minimal_training_results.json`
  - `benchmark/autodiff/results/minimal_training_report.md`

## Build And Run

### Configure and build

```bash
/Users/hhd/miniforge3/envs/py310/bin/cmake -S /Users/hhd/Desktop/test/c-pinn -B /Users/hhd/Desktop/test/c-pinn/build -DPINN_USE_TORCH=OFF -DCMAKE_BUILD_TYPE=Release
make -C /Users/hhd/Desktop/test/c-pinn/build -j"$(sysctl -n hw.ncpu)" example_autodiff_nested_graph example_autodiff_matrix_graph unit_tests
```

### Run example (human-readable)

```bash
/Users/hhd/Desktop/test/c-pinn/build/examples/example_autodiff_nested_graph
```

### Run example (machine-readable JSON)

```bash
/Users/hhd/Desktop/test/c-pinn/build/examples/example_autodiff_nested_graph 0.5 --json
```

### Run matrix-style AD mini PINN prototype

```bash
/Users/hhd/Desktop/test/c-pinn/build/examples/example_autodiff_matrix_graph --iters 25 --samples 6
```

### Run matrix-style AD JSON mode

```bash
/Users/hhd/Desktop/test/c-pinn/build/examples/example_autodiff_matrix_graph --json --equation all --iters 25 --samples 6
```

### Run detailed tests

```bash
cd /Users/hhd/Desktop/test/c-pinn
python3 benchmark/autodiff/test_nested_graph.py
```

### Run stress test for scalar nested-graph (tight threshold)

```bash
cd /Users/hhd/Desktop/test/c-pinn
python3 benchmark/autodiff/test_nested_graph.py --random-cases 1000 --tol-d1 1e-13 --tol-d2 1e-13 --tol-gap 1e-13
```

### Run matrix AD + legacy 3-equation regression test

```bash
cd /Users/hhd/Desktop/test/c-pinn
python3 benchmark/autodiff/test_matrix_graph.py
```

Default stress criterion in this script is `--stress-min-pass-rate 0.8`.
Use `--stress-min-pass-rate 1.0` only when you explicitly want the stricter all-seed pass requirement.

### Run minimal Node training closure test

```bash
cd /Users/hhd/Desktop/test/c-pinn
python3 benchmark/autodiff/test_minimal_training.py
```

### Run regression unit tests

```bash
/Users/hhd/Desktop/test/c-pinn/build/tests/unit_tests
```

## Detailed Test Status (2026-04-01)

Source: `benchmark/autodiff/results/nested_graph_test_report.md`

- Scalar nested-graph pressure test verdict: PASS
  - Grid tests: 13 cases
  - Random tests: 1000 cases
  - Thresholds:
    - `abs_err_d1 <= 1e-13`
    - `abs_err_d2 <= 1e-13`
    - `|dy_dx_graph - dy_dx_detached| <= 1e-13`
    - `dy_dx_graph.requires_grad == true`
    - `g1_nodes > g0_nodes` and `g1_edges > g0_edges`
  - Observed maxima:
    - Grid `max_abs_err_d1 = 1.0e-15`
    - Grid `max_abs_err_d2 = 0.0`
    - Random `max_abs_err_d1 = 2.0e-15`
    - Random `max_abs_err_d2 = 4.0e-15`
    - `max_gap_graph_vs_detached = 0.0`

- Matrix AD mini PINN regression verdict: PASS
  - Config: `iters=25`, `samples=6`, `layers=[2,4,1]`, `lr=5e-4`, `grad_clip=50`
  - KdV residual loss: `0.390336 -> 0.206351` (ratio `0.528648`)
  - Sine-Gordon residual loss: `6.993313e-4 -> 6.741752e-4` (ratio `0.964028`)
  - Allen-Cahn residual loss: `0.125430 -> 0.037572` (ratio `0.299546`)
  - `nan_grad_count = 0` for all three equations
  - Multi-seed stress default uses `pass_rate >= 0.8` as the pass criterion for stochastic mini-PINN validation

- Legacy pure C 3-equation regression verdict: PASS
  - KdV loss: `iter0 9.65526e-2 -> iter900 1.53759e-7`
  - Sine-Gordon loss: `iter0 6.66129e-2 -> iter900 2.63194e-7`
  - Allen-Cahn loss: `iter0 9.43103e-1 -> iter900 4.57256e-6`

## Implementation Notes

- `grad(output, input, create_graph=false)` returns detached gradient values.
- `grad(output, input, create_graph=true)` builds a gradient graph that supports higher-order gradients.
- Demo function in the prototype: `y = x^3 + sin(x)`.
- Demonstrated derivatives:
  - first derivative: `dy/dx = 3x^2 + cos(x)`
  - second derivative: `d2y/dx2 = 6x - sin(x)`
- Matrix-style prototype uses scalar nodes as matrix elements and composes:
  - `matmul`
  - row-wise bias add
  - `tanh`
  - nested `grad(..., create_graph=true)` for PDE derivatives (`u_t`, `u_x`, `u_xx`, `u_xxx`, `u_tt`)
- Matrix-style prototype is intended for mechanism validation and testability, not speed.

## Environment Notes

- In this workspace, `cmake` may not be in shell `PATH`.
- Use explicit cmake binary path shown above when reconfiguring build files.

## Current Limits

- Matrix prototype is still scalar-node backed (element-wise graph), so performance is not representative of optimized tensor AD engines.
- Prototype is not yet integrated into the full PINN training path (`src/nn` + PDE residual stack).
- No tensor-level operator fusion or BLAS-backed AD engine in this prototype.
