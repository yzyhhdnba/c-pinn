# Pure C vs Torch-Only Modules

## 1. 项目目标与现状

本项目旨在逐步移除对 LibTorch 的依赖，转而使用纯 C++ (C++17) 实现核心功能。目标是构建一个轻量级、无外部重依赖（仅依赖 Eigen/JSON）的 PINN 求解器。

**当前状态 (2025-12-23)**：
- **核心层 (Core/NN/Geometry)**：已完全实现纯 C 版本，支持前向推理、模型保存/加载、几何采样。
- **训练层 (Training/Autograd)**：仍依赖 LibTorch 进行自动微分和梯度更新。
- **构建系统**：支持 `PINN_USE_TORCH=OFF` 模式，此时仅编译纯 C 模块（适用于推理环境）。

---

## 2. 功能矩阵对比

| 模块 | 功能 | Pure C (PINN_USE_TORCH=OFF) | Torch-Only (PINN_USE_TORCH=ON) |
| :--- | :--- | :--- | :--- |
| **Tensor** | 基础运算 (+-*/) | ✅ `pinn::core::Tensor` | ✅ `torch::Tensor` |
| | 形状操作 (reshape/slice) | ✅ 支持 | ✅ 支持 |
| | 自动微分 (Autograd) | ❌ 不支持 | ✅ 支持 |
| | GPU 加速 | ❌ CPU Only | ✅ CUDA 支持 |
| **Neural Network** | FNN (全连接) | ✅ `pinn::nn::Fnn` | ✅ (兼容) |
| | ResNet/CNN/Transformer | ❌ 暂不支持 | ✅ `torch::nn::*` |
| | 激活函数 (Tanh/ReLU/Sin) | ✅ 手写实现 (OpenMP) | ✅ LibTorch 后端 |
| | 初始化 (Xavier/Kaiming) | ✅ `pinn::core::Rng` | ✅ `torch::nn::init` |
| **Training** | 优化器 (Adam/LBFGS) | ⚠️ 仅算法实现，无梯度接口 | ✅ `torch::optim` |
| | 损失函数 (MSE) | ✅ `pinn::core::mse_loss` | ✅ `torch::mse_loss` |
| | 梯度计算 | ❌ 需手动实现 | ✅ `loss.backward()` |
| **Utils** | Checkpoint (保存/加载) | ✅ 二进制格式 (`.bin`) | ✅ Torch 序列化 (`.pt`) |
| | 可视化 | ❌ | ✅ `VisualizationCallback` |

---

## 3. 详细模块说明

### ✅ 纯 C 模块 (Core Modules)

这些模块不依赖任何 Torch 头文件，可在任何 C++17 环境编译。

#### 3.1 核心张量 (pinn::core::Tensor)
- **实现**：基于 `std::vector<double>` 的扁平连续存储 (Row-Major)。
- **功能**：
  - 算术运算：`add`, `sub`, `mul`, `div`, `pow`, `abs`, `sqrt`, `sin`, `tanh`, `relu`。
  - 规约：`sum`, `mean`, `norm`。
  - 形状：`reshape`, `transpose`, `slice`, `unsqueeze`, `squeeze`, `cat`, `stack`。
  - 工厂：`zeros`, `ones`, `full`, `zeros_like`, `ones_like`, `arange`, `linspace`, `rand`, `randn`。
- **并行**：主要算子使用 OpenMP 进行多线程加速。

#### 3.2 神经网络 (pinn::nn)
- **Linear**：纯 C 全连接层，使用 `gemm` (Tiled Matrix Multiplication) 实现。
- **Fnn**：多层感知机容器，管理层与激活函数。
- **Optimizer**：
  - `AdamOptimizer`：封装了纯 C 的 `gcn_update_adam` 算法。
  - `LbfgsOptimizer`：封装了纯 C 的 `lbfgs_step` 算法。
  - *注：目前优化器类仅提供算法实现，因缺乏自动微分，无法自动获取梯度。*

#### 3.3 工具 (pinn::utils)
- **CheckpointManagerC**：自定义二进制格式的模型保存与加载。
  - 文件头：Magic Number (`PINN`) + Version。
  - 结构：层数、每层维度。
  - 数据：扁平权重的二进制流。

---

### 🔴 Torch 依赖模块 (Torch-Dependent Modules)

这些模块必须在 `PINN_USE_TORCH=ON` 时才能编译。

#### 3.4 自动微分与损失 (pinn::loss)
- **compute_gradients / compute_hessian**：直接调用 `torch::autograd::grad` 计算一阶/二阶导数。
- **compute_pde_residual**：构建计算图以求解 PDE 残差。

#### 3.5 训练器 (pinn::model::Trainer)
- 管理训练循环、Epoch 迭代。
- 负责调用 `optimizer->step()` 和 `loss.backward()`。
- 处理 RAR (Residual-Adaptive Refinement) 采样策略。

---

## 4. 代码示例

### 4.1 Pure C 模式：构建网络与推理

```cpp
#include "pinn/nn/fnn.hpp"
#include "pinn/core/tensor.hpp"
#include "pinn/utils/checkpoint_c.hpp"

using namespace pinn;

int main() {
    // 1. 定义网络架构
    std::vector<int> layers = {2, 50, 50, 50, 1};
    auto activation = nn::ActivationFn::get("tanh");
    
    // 2. 初始化网络 (使用纯 C 随机数生成器)
    nn::Fnn net(layers, activation, nn::InitType::kXavierUniform, 0.0, 42);

    // 3. 准备输入数据 (Batch=10, Dim=2)
    auto input = core::Tensor::rand_uniform({10, 2});

    // 4. 前向传播
    auto output = net.forward(input);

    // 5. 保存模型 (二进制格式)
    utils::CheckpointManagerC ckpt("checkpoints");
    ckpt.save(net, 0, 0.123);
    
    return 0;
}
```

### 4.2 Torch 模式：训练循环 (伪代码)

```cpp
// 仅在 PINN_USE_TORCH=ON 时可用
#include "pinn/model/trainer.hpp"

void train() {
    // ... 初始化 Model 和 Trainer ...
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 1. 前向传播
        auto loss = compute_losses(...);
        
        // 2. 反向传播 (LibTorch)
        optimizer->zero_grad();
        loss.total_loss.backward();
        
        // 3. 优化器更新
        optimizer->step();
    }
}
```

---

## 5. 构建指南

### 5.1 纯 C 模式 (推荐用于推理/部署)

此模式不需要安装 LibTorch，编译速度快，产物小。

```bash
# 1. 配置 (关闭 Torch)
cmake -S . -B build_pure -DPINN_USE_TORCH=OFF -DCMAKE_BUILD_TYPE=Release

# 2. 编译
cmake --build build_pure -j

# 3. 运行测试
ctest --test-dir build_pure
```

### 5.2 Torch 模式 (用于训练/开发)

需要预先下载并解压 LibTorch。

```bash
# 1. 设置 LibTorch 路径
export CMAKE_PREFIX_PATH=/path/to/libtorch

# 2. 配置 (开启 Torch)
cmake -S . -B build_torch -DPINN_USE_TORCH=ON -DCMAKE_BUILD_TYPE=Release

# 3. 编译
cmake --build build_torch -j
```

---

## 6. 目录结构解析

```text
include/pinn/
├── core/           # [Pure C] 核心张量、随机数
├── geometry/       # [Pure C] 几何域、采样
├── nn/             # [Pure C] 神经网络层、激活、初始化、优化器算法
├── utils/          # [Mixed]  工具类
│   ├── checkpoint_c.hpp  # [Pure C] 二进制检查点
│   └── checkpoint.hpp    # [Torch]  Torch 序列化检查点
├── loss/           # [Torch]  损失函数、自动微分
├── pde/            # [Mixed]  PDE 定义 (部分依赖 Autograd)
└── model/          # [Torch]  模型容器、训练器
```

---

## 7. 下一步计划

为了完全移除 Torch 依赖，我们需要解决 **自动微分 (Autograd)** 这一核心难题。

1.  **短期**：完善 `AdamOptimizer` 和 `LbfgsOptimizer` 的纯 C 接口，使其能接收外部计算好的梯度（例如通过有限差分计算的梯度）进行参数更新。
2.  **中期**：引入轻量级自动微分库（如 `autodiff` 或 `Eigen::AutoDiff`），或者为 FNN 实现手写的反向传播算法。
3.  **长期**：实现完整的计算图引擎，支持任意 PDE 残差的自动微分。

