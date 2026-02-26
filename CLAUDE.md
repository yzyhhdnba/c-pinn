# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

C-PINN is a C++17 Physics-Informed Neural Network (PINN) framework with **dual-mode compilation**:
- **Pure C++ mode** (`PINN_USE_TORCH=OFF`): No deep learning framework dependencies, manual backprop, finite differences for derivatives
- **LibTorch mode** (`PINN_USE_TORCH=ON`): PyTorch backend with autograd, GPU support

## Build Commands

### Pure C++ Mode (Recommended for development)
```bash
# Configure
cmake -S .. -B . -DPINN_USE_TORCH=OFF -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(sysctl -n hw.ncpu)

# Run tests
ctest --output-on-failure

# Run specific example
./examples/example_pure_c_kdv
./examples/example_pure_c_sine_gordon
./examples/example_pure_c_allen_cahn
./examples/example_pure_c_inference
```

### LibTorch Mode
```bash
# Set LIBTORCH_PATH to your LibTorch installation
export LIBTORCH_PATH=/path/to/libtorch

# Configure with Torch enabled
cmake -S .. -B . -DPINN_USE_TORCH=ON -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH

# Build
cmake --build . -j
```

## Architecture

### Dual-Mode Structure

| Component | Pure C++ | LibTorch |
|-----------|----------|----------|
| Tensor | `pinn::core::Tensor` | `torch::Tensor` |
| Network | `pinn::nn::Fnn` | `torch::nn::Module` |
| Optimizer | `pinn::nn::AdamOptimizer` | `torch::optim::Optimizer` |
| Loss | Manual computation | `torch::nn::*Loss` |
| Checkpoint | `CheckpointManagerC` (binary) | `CheckpointManager` |

### Key Namespaces
- `pinn::core`: Tensor, Rng (random number generation)
- `pinn::nn`: Fnn, Linear, optimizers, activations
- `pinn::geometry`: Interval, Rectangle, sampling strategies
- `pinn::pde`: PDE definitions, boundary conditions
- `pinn::utils`: Checkpointing, logging, config

### Key Types
- `pinn::Tensor` = `pinn::core::Tensor`
- `pinn::Scalar` = `double`
- `pinn::nn::TYPE_VAL` = `float`

### Pure C++ Training Pattern

Since there's no autograd, manual backpropagation is required:

```cpp
// 1. Forward pass (caches inputs for backward)
Tensor output = net.forward(input);

// 2. Compute loss and gradients manually
Tensor loss = residual.pow(2.0).mean_all();
Tensor dL_dresidual = residual * (2.0 / batch_size);

// 3. Backward pass (requires re-running forward to set cache per point)
net.forward(point);
net.backward(gradient_for_point);

// 4. Update weights
optimizer.step();
```

### Derivative Computation (Finite Difference)

For PDE terms, derivatives are computed using finite differences:

```cpp
// Central difference for first derivative
// f'(x) ≈ (f(x+h) - f(x-h)) / 2h

// Third derivative
// f'''(x) ≈ (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (2h³)
```

See [pure_c_kdv.cpp](examples/pure_c_kdv.cpp) for complete example with KdV equation.

## Code Patterns

### Activation Functions
- Defined as `std::function<Tensor(const Tensor&)>` in `activation.hpp`
- Available: "tanh", "relu", "sin" (with derivatives)
- Use `activation_from_string()` and `activation_derivative_from_string()`

### Network Initialization
```cpp
// Xavier Uniform (default for tanh)
nn::Fnn net(layers, "tanh", nn::InitType::kXavierUniform, 0.0, seed);

// Kaiming He (recommended for relu)
nn::Fnn net(layers, "relu", nn::InitType::kKaimingUniform, 0.0, seed);
```

### Tensor Creation
```cpp
// Factory methods
Tensor::zeros({batch, dim});
Tensor::ones({batch, dim});
Tensor::rand_uniform({batch, dim}, seed);
Tensor::randn({batch, dim}, seed);
Tensor::linspace(start, end, steps);
```

### Checkpointing (Pure C++)
```cpp
CheckpointManagerC ckpt("checkpoints/", save_every);
ckpt.save(net, epoch, loss);
ckpt.load_latest(net);
```
Binary format stores layer sizes + flattened weights.

## Testing

The test suite is in `tests/tensor_test.cpp`. Run with:
```bash
./tests/unit_tests
# or
ctest --test-dir build_dir -V
```
