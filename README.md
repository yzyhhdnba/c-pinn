# C-PINN: C++17 物理信息神经网络框架

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.22+-green.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**C-PINN** 是一个完全用 C++17 实现的物理信息神经网络（Physics-Informed Neural Networks）框架，无需深度学习框架依赖，支持双模编译。

## 🎯 核心特点

- ✅ **纯 C++17 实现** — 不依赖 Python、TensorFlow、PyTorch（LibTorch 为可选扩展）
- ✅ **双模编译** — 纯 C++ 模式（零外部 DL 框架）与 LibTorch 混合模式
- ✅ **有限差分 + 手动反向传播** — 不依赖 Autograd 完成 PINN 训练
- ✅ **最小自动微分原型** — 新增标量嵌套图与矩阵 `matmul` 原型，用于模拟 `create_graph=true`
- ✅ **手写 Adam 优化器** — 包含动量、偏差修正、可选权重衰减
- ✅ **分层验证体系** — 解析导数精度、压力测试、多 seed 稳定性、legacy 三方程回归

## 📌 最新进展（2026-04）

- 新增标量嵌套图 AD 示例：`examples/autodiff_nested_graph.cpp`
- 新增矩阵 AD + `matmul` 示例：`examples/autodiff_matrix_graph.cpp`
- 新增细致测试脚本：`benchmark/test_autodiff_nested_graph.py`
- 新增矩阵 AD + legacy 回归脚本：`benchmark/test_autodiff_matrix_graph.py`
- 新增技术报告：`benchmark/autodiff_nested_graph_tech_report.md`

当前默认训练主路径仍是“有限差分 + 手动反向传播”（`example_pure_c_*`）；
新增 AD 原型位于 `examples/`，用于机制验证与后续集成设计，不直接替代生产训练路径。

---

## 📊 架构模式

项目通过 CMake 选项 `PINN_USE_TORCH` 控制编译模式：

| 特性 | **纯 C++ 模式** (`OFF`) | **LibTorch 模式** (`ON`) |
|:-----|:-----------------------|:----------------------|
| **编程语言** | C++17 | C++17 |
| **外部依赖** | nlohmann_json 仅此一项 | + LibTorch (PyTorch C++ API) |
| **微分方式** | 有限差分 + 手动反向传播 | LibTorch Autograd |
| **实验性 AD 原型** | 提供（`examples/autodiff_*`） | N/A |
| **硬件支持** | CPU (OpenMP 加速) | CPU / CUDA GPU |
| **部署场景** | 嵌入式、边缘计算、推理 | GPU 训练、研究原型 |
| **编译产物** | 独立可执行文件 (150K-250K) | 依赖 LibTorch 动态库 |

**注意**：虽然文件扩展名为 `.cpp/.hpp`，但纯 C++ 模式中的代码是标准 C++17，使用 C++ 特性如类、模板、STL 容器等。"纯 C++" 指**不依赖外部深度学习框架**，而非使用 C 语言。

---

## 🏗️ 核心组件（纯 C++ 模式）

### 1. 张量引擎 (`pinn::core::Tensor`)

自研轻量级张量库，无第三方依赖：

```cpp
// 特性
class Tensor {
    // 存储：Row-Major 连续内存 (std::vector<double>)
    // 支持：动态形状、多维索引、广播、切片
    // 算子：+, -, *, /, matmul, pow, sin, tanh, relu
    // 归约：sum, mean, norm
};

// 创建示例
auto x = core::Tensor::rand_uniform({64, 2});  // [batch=64, dim=2]
auto w = core::Tensor::zeros({2, 50});         // 权重矩阵
auto y = x.matmul(w);                          // 前向传播
auto loss = y.pow(2.0).mean_all();             // MSE 损失
```

**实现细节**：
- Row-Major 内存布局（缓存友好）
- 显式循环展开（无虚函数调用开销）
- OpenMP 并行加速（GEMM、逐元素运算）

### 2. 神经网络模块 (`pinn::nn`)

#### 全连接层 (`Linear`)

```cpp
class Linear {
public:
    // 前向：y = xW^T + b
    Tensor forward(const Tensor& x) const;
    
    // 反向：手动链式法则
    // grad_output: dL/dy [batch, out_features]
    // 返回: dL/dx [batch, in_features]
    Tensor backward(const Tensor& grad_output, const Tensor& input);
    
private:
    Tensor weight_t_;      // [in_features, out_features]
    Tensor bias_;          // [out_features]
    Tensor grad_weight_;   // 累积梯度
    Tensor grad_bias_;
};
```

#### 多层感知机 (`Fnn`)

```cpp
// 构建 2 → 50 → 50 → 50 → 1 网络
std::vector<int> layers = {2, 50, 50, 50, 1};
nn::Fnn net(layers, "tanh", nn::InitType::kXavierUniform, 0.0, /*seed=*/42);

// 训练流程
Tensor output = net.forward(input);           // 缓存中间激活
Tensor grad_in = net.backward(grad_output);  // 反向传播
```

#### 激活函数

| 函数 | 前向 | 反向 |
|:-----|:-----|:-----|
| `tanh` | $\tanh(x)$ | $1 - \tanh^2(x)$ |
| `relu` | $\max(0, x)$ | $x > 0 ? 1 : 0$ |
| `sin` | $\sin(x)$ | $\cos(x)$ |

所有激活函数及其导数均手动实现，无自动微分依赖。

### 3. 优化器 (`pinn::nn::AdamOptimizer`)

完整 Adam 算法实现（参考 [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)）：

```cpp
// 配置
nn::AdamOptimizer::Options opts;
opts.lr = 1e-3;
opts.beta1 = 0.9;    // 一阶矩估计指数衰减率
opts.beta2 = 0.999;  // 二阶矩估计指数衰减率
opts.epsilon = 1e-8;
opts.weight_decay = 0.0;

nn::AdamOptimizer optimizer(net, opts);

// 训练步
optimizer.zero_grad();
// ... 计算损失并调用 net.backward() ...
optimizer.step();  // 更新所有参数
```

**实现要点**：
- 动量缓冲区：一阶矩 $m_t$、二阶矩 $v_t$（使用 `malloc/free` 管理）
- 偏差修正：$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$
- 权重衰减：L2 正则化可选

### 4. 随机数生成 (`pinn::core::Rng`)

确定性伪随机数生成器（基于 SplitMix64）：

```cpp
core::Rng rng(42);  // 固定种子

double u = rng.uniform01();          // U(0, 1)
double n = rng.normal01();           // N(0, 1)，Box-Muller 变换
int64_t i = rng.randint(0, 100);     // [0, 100) 均匀整数
```

### 5. 模型持久化 (`pinn::utils::CheckpointManagerC`)

自定义二进制格式（`.bin`），比 PyTorch `.pt` 文件更紧凑：

```cpp
// 保存
CheckpointManagerC ckpt("./checkpoints", /*save_every=*/100);
ckpt.save(net, epoch, loss);  
// 生成文件：epoch_500_loss_1.23e-07.bin

// 加载
ckpt.load_latest(net);  // 自动找到最新检查点
```

**文件格式**：
```
[4 bytes] Magic: 0x50494E4E ('PINN')
[4 bytes] Version: 1
[4 bytes] Num layers
[4 bytes] Input dim
[4*N bytes] Layer dims
[Variable] Weights (扁平化 double 数组)
[Variable] Biases
```

---

## 🧮 PINN 训练原理

### 微分策略：有限差分 + 手动反向传播

纯 C++ 模式采用**混合微分策略**：

#### 1. PDE 导数：有限差分（Finite Difference）

用于计算偏微分方程中的空间/时间导数：

| 导数 | 公式 | 模板点数 |
|:-----|:-----|:--------|
| $u_x$ | $\frac{u(x+h) - u(x-h)}{2h}$ | 3 |
| $u_{xx}$ | $\frac{u(x+h) - 2u(x) + u(x-h)}{h^2}$ | 3 |
| $u_{xxx}$ | $\frac{u(x+2h) - 2u(x+h) + 2u(x-h) - u(x-2h)}{2h^3}$ | 5 |

#### 2. 参数梯度：手动反向传播

用于更新神经网络权重：

```cpp
// 链式法则展开
// dL/dW_i = dL/da_{i+1} × da_{i+1}/dz_i × dz_i/dW_i
//                      ↑ 激活函数导数  ↑ 输入 x^T

Tensor Linear::backward(const Tensor& grad_output, const Tensor& input) {
    // grad_output: dL/d(output) [batch, out_features]
    // input: 前向传播时缓存的输入 [batch, in_features]
    
    // 计算 dL/dW = input^T × grad_output
    grad_weight_ = input.transpose(0, 1).matmul(grad_output);
    
    // 计算 dL/db = sum(grad_output, dim=0)
    grad_bias_ = grad_output.sum(0);
    
    // 传递给前一层：dL/d(input) = grad_output × W
    return grad_output.matmul(weight_t_.transpose(0, 1));
}
```

### 完整训练流程（KdV 方程示例）

**目标**：求解 Korteweg–de Vries 方程 $u_t + 6uu_x + u_{xxx} = 0$

```cpp
for (int iter = 0; iter < 1000; ++iter) {
    optimizer.zero_grad();
    
    // 1. 采样配置点
    auto x_t = core::Tensor::rand_uniform({64, 2});  // [batch, (x, t)]
    
    // 2. 构造有限差分模板
    std::vector<StencilPoint> stencils;
    stencils.push_back({x_t.clone(), 0.0});                    // 中心点
    stencils.push_back({shift(x_t, 0, +h), 0.0});              // x+h
    stencils.push_back({shift(x_t, 0, -h), 0.0});              // x-h
    stencils.push_back({shift(x_t, 0, +2*h), 0.0});            // x+2h
    stencils.push_back({shift(x_t, 0, -2*h), 0.0});            // x-2h
    stencils.push_back({shift(x_t, 1, +h), 0.0});              // t+h
    stencils.push_back({shift(x_t, 1, -h), 0.0});              // t-h
    
    // 3. 前向传播所有模板点
    for (auto& sp : stencils) {
        sp.output = net.forward(sp.input);
    }
    
    // 4. 计算有限差分导数
    auto u = stencils[0].output;
    auto u_x = (stencils[1].output - stencils[2].output) / (2*h);
    auto u_xxx = (stencils[3].output - 2*stencils[1].output 
                + 2*stencils[2].output - stencils[4].output) / (2*h*h*h);
    auto u_t = (stencils[5].output - stencils[6].output) / (2*h);
    
    // 5. PDE 残差
    auto residual = u_t + 6.0 * u * u_x + u_xxx;
    auto loss = residual.pow(2.0).mean_all();
    
    // 6. 反向传播（手动链式法则）
    auto dL_dR = residual * (2.0 / batch_size);
    
    // 对每个模板点：dL/du_i = dL/dR × dR/du_i
    net.forward(stencils[0].input);
    net.backward(dL_dR * 6.0 * u_x);  // dR/du = 6*u_x
    
    net.forward(stencils[1].input);
    net.backward(dL_dR * (6.0*u/(2*h) - 2.0/(2*h*h*h)));  // dR/du_xp
    // ... 其余模板点类似 ...
    
    // 7. 参数更新
    optimizer.step();
}
```

---

## 🚀 快速开始

### 系统要求

**必需**（纯 C++ 模式）：
- **编译器**：GCC ≥ 9 / Clang ≥ 11 / MSVC ≥ 19.28（支持 C++17）
- **CMake**：≥ 3.22
- **依赖库**：[nlohmann/json](https://github.com/nlohmann/json)（JSON 配置解析）

**可选**：
- **LibTorch**：≥ 2.0（仅 Torch 模式需要）
- **OpenMP**：多线程加速（推荐）

### 编译步骤

#### 方式 A：纯 C++ 模式（推荐）

```bash
# macOS
brew install cmake nlohmann-json libomp

# Ubuntu/Debian
sudo apt install cmake nlohmann-json3-dev libomp-dev

# 编译
mkdir build && cd build
cmake .. -DPINN_USE_TORCH=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

> 如果当前 shell 找不到 `cmake`，可用绝对路径（本仓库已验证）：
> `/Users/hhd/miniforge3/envs/py310/bin/cmake`

#### 方式 B：LibTorch 模式

```bash
# 下载 LibTorch (CPU 版本示例)
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.1.zip
unzip libtorch-shared-with-deps-2.5.1.zip

# 编译
mkdir build_torch && cd build_torch
cmake .. -DPINN_USE_TORCH=ON -DCMAKE_PREFIX_PATH=$(pwd)/../libtorch
cmake --build . -j$(nproc)
```

### 运行示例

```bash
cd build

# 纯 C++ 示例
./examples/example_pure_c_kdv            # KdV 方程训练
./examples/example_pure_c_sine_gordon    # Sine-Gordon 方程
./examples/example_pure_c_allen_cahn     # Allen-Cahn 方程
./examples/example_pure_c_inference      # 基础推理测试

# 最小 AD 原型（标量嵌套图）
./examples/example_autodiff_nested_graph
./examples/example_autodiff_nested_graph 0.5 --json

# 最小 AD 原型（矩阵 + matmul）
./examples/example_autodiff_matrix_graph --iters 25 --samples 6
./examples/example_autodiff_matrix_graph --json --equation all --iters 25 --samples 6

# LibTorch 示例（需启用 PINN_USE_TORCH）
./examples/example_burgers               # Burgers 方程（使用 Autograd）
```

### 运行自动化测试

```bash
# 标量嵌套图细致测试
python3 benchmark/test_autodiff_nested_graph.py

# 标量嵌套图压力测试（1000 随机点 + 严格阈值）
python3 benchmark/test_autodiff_nested_graph.py --random-cases 1000 --tol-d1 1e-13 --tol-d2 1e-13 --tol-gap 1e-13

# 矩阵 AD + legacy 三方程回归
python3 benchmark/test_autodiff_matrix_graph.py

# 矩阵 AD 多 seed 压测（pass_rate 口径）
python3 benchmark/test_autodiff_matrix_graph.py --stress-seeds 7,17,27,37,47 --stress-iters 80 --stress-samples 16 --stress-min-pass-rate 0.8
```

**预期输出**（KdV 方程，1000 次迭代）：
```
=== Pure C++ KdV Solver (Stencil Framework) ===
Iter 0 Loss: 0.0965526
Iter 100 Loss: 4.74016e-06
Iter 200 Loss: 5.31217e-07
...
Iter 900 Loss: 1.53759e-07
Training finished.
```

---

## 📁 项目结构

```
c-pinn/
├── include/pinn/           # 公共头文件
│   ├── core/               # [纯 C++] 核心数据结构
│   │   ├── tensor.hpp      # 张量类（存储、算子、形状操作）
│   │   └── rng.hpp         # 随机数生成器（SplitMix64 + Box-Muller）
│   ├── nn/                 # [纯 C++] 神经网络
│   │   ├── fnn.hpp         # 全连接网络（Linear + Fnn）
│   │   ├── optimizer.hpp   # Adam / L-BFGS 优化器
│   │   ├── activation.hpp  # 激活函数（tanh, relu, sin）
│   │   ├── initialization.hpp # 权重初始化（Xavier, Kaiming）
│   │   ├── gemm.hpp        # 矩阵乘法（OpenMP 并行）
│   │   └── adam.hpp        # Adam 算法实现
│   ├── geometry/           # [纯 C++] 几何域与采样
│   │   ├── interval.hpp    # 1D 区间
│   │   ├── rectangle.hpp   # N维超矩形
│   │   ├── difference.hpp  # 几何差集
│   │   └── sampling.hpp    # 采样策略（均匀网格、拉丁超立方）
│   ├── utils/              # 工具类
│   │   ├── checkpoint_c.hpp # [纯 C++] 二进制模型存储
│   │   ├── stencil.hpp     # 有限差分模板点管理
│   │   ├── logger.hpp      # 日志
│   │   └── config.hpp      # JSON 配置解析
│   ├── pde/                # [混合] PDE 定义
│   │   ├── pde.hpp         # PDE 抽象类
│   │   ├── boundary_condition.hpp # 边界条件（Dirichlet, Neumann, Periodic）
│   │   └── parser.hpp      # PDE 表达式解析器
│   ├── loss/               # [仅 Torch] 损失函数
│   │   └── loss_terms.hpp
│   └── model/              # [仅 Torch] 训练器
│       ├── model.hpp
│       └── trainer.hpp
├── src/                    # 实现文件
│   ├── core/tensor.cpp
│   ├── nn/{fnn, optimizer, adam, ...}.cpp
│   ├── geometry/{interval, rectangle, ...}.cpp
│   └── utils/{checkpoint_c, logger, ...}.cpp
├── examples/               # 示例程序
│   ├── pure_c_kdv.cpp      # [纯 C++] KdV 方程
│   ├── pure_c_sine_gordon.cpp # [纯 C++] Sine-Gordon 方程
│   ├── pure_c_allen_cahn.cpp  # [纯 C++] Allen-Cahn 方程
│   ├── pure_c_inference.cpp   # [纯 C++] 推理示例
│   ├── autodiff_nested_graph.cpp # [纯 C++] 标量嵌套图 AD 原型
│   ├── autodiff_matrix_graph.cpp # [纯 C++] 矩阵 AD + matmul 原型
│   └── burgers.cpp         # [Torch] Burgers 方程
├── benchmark/              # 基准与验证
│   ├── test_autodiff_nested_graph.py
│   ├── test_autodiff_matrix_graph.py
│   ├── autodiff_nested_graph_tech_report.md
│   ├── autodiff_nested_graph_test_report.md
│   └── autodiff_matrix_graph_test_report.md
├── tests/                  # 单元测试
│   └── tensor_test.cpp
├── config/                 # 配置文件
│   └── pinn_config.json
├── docs/                   # 文档
│   ├── pure_c_vs_torch.md  # 技术对比
│   └── requirements.md
├── CMakeLists.txt          # 主 CMake 配置
└── README.md
```

---

## 🔬 验证结果（截至 2026-04）

**测试环境**：Apple M4, 16GB RAM, Apple Clang 17.0.0, Release 模式

### A. Legacy 纯 C++ 三方程收敛

| 方程 | PDE 形式 | 初始损失 | 最终损失 (iter 900) | 实测耗时 |
|:-----|:--------|:--------|:-------------------|:---------|
| **KdV** | $u_t + 6uu_x + u_{xxx} = 0$ | 0.0966 | 1.54×10⁻⁷ | 8.7s |
| **Sine-Gordon** | $u_{tt} - u_{xx} + \sin(u) = 0$ | 0.0666 | 2.63×10⁻⁷ | 6.2s |
| **Allen-Cahn** | $u_t - 0.0001u_{xx} + 5(u^3-u) = 0$ | 0.943 | 4.57×10⁻⁶ | 6.2s |

**训练配置**：网络 [2,50,50,50,1]、batch_size=64、Adam(lr=1e-3)、tanh 激活、h=1e-3、1000 次迭代。

### B. 标量嵌套图 AD 精度与图结构

来源：`benchmark/autodiff_nested_graph_test_results.json`

- 阈值：`tol_d1=1e-13`、`tol_d2=1e-13`、`tol_gap=1e-13`
- 网格 13 点 + 随机 1000 点全部通过
- 随机集最大误差：`max_abs_err_d1=2e-15`、`max_abs_err_d2=4e-15`
- 图性质验证通过：`dy_dx_graph.requires_grad=true` 且 `g1_nodes/g1_edges > g0_nodes/g0_edges`

### C. 矩阵 AD（含 matmul）与多 seed 压测

来源：`benchmark/autodiff_matrix_graph_test_results.json`

- Smoke 配置（`iters=25, samples=6`）下，KdV/Sine-Gordon/Allen-Cahn 的残差 loss 均下降，且 `nan_grad_count=0`
- 压测配置（`iters=80, samples=16, seeds=[7,17,27,37,47]`）
    - Allen-Cahn：pass_rate=1.00
    - KdV：pass_rate=0.80
    - Sine-Gordon：pass_rate=0.80
    - 在 `min_pass_rate=0.8` 判定口径下，整体为 PASS

**可执行文件大小**：152K-246K（静态链接 libpinn.a，不含 LibTorch）。

### D. 当前已知限制（AD 原型）

- 矩阵 AD 仍为“标量节点拼矩阵”的机制验证实现，不代表高性能张量引擎。
- 尚未集成到 `src/nn` + PDE 主训练路径。
- 报告中已记录两个工程风险：
    - `tanh` 的 VJP 闭包潜在引用环（长跑内存风险）
    - `--equation all` 下 seed 报告与 effective seed 偏移需明确

---

## 🧪 技术细节

### C++ 特性使用清单

虽然称为"纯 C++"，项目实际大量使用现代 C++17 特性：

| 特性类别 | 使用情况 |
|:---------|:--------|
| **面向对象** | 类（`class Tensor`）、继承（`AdamOptimizer : public Optimizer`）、虚函数 |
| **模板** | `template<typename T> data_ptr()`、泛型编程 |
| **STL 容器** | `std::vector`、`std::string`、`std::pair`、`std::function` |
| **现代特性** | `auto`、lambda 表达式、`if constexpr`（C++17）、RAII |
| **标准库** | `<algorithm>`、`<cmath>`、`<filesystem>`、`<regex>` |
| **异常处理** | `throw std::runtime_error(...)` |

以上特性均为 C++ 特有，标准 C 语言不支持。项目中的「纯 C」仅指不依赖外部深度学习框架。

### 实现特点

1. **内存布局**：Row-Major 连续存储（`std::vector<std::byte>` 底层缓冲）
2. **GEMM**：手写三重循环矩阵乘法，支持 OpenMP 并行（`#pragma omp parallel for`），可选 CBLAS 后端
3. **激活函数**：通过 `std::function` 存储（存在间接调用开销），运行时从字符串映射选择
4. **优化器内存**：Adam 一阶/二阶矩使用 `malloc/free` 管理扁平数组
5. **张量内存**：RAII 管理（`std::vector<std::byte>` 自动释放）
6. **虚函数**：`Optimizer` 基类使用虚函数分派（`step()`、`zero_grad()` 等）

### 与 PyTorch 方案的区别

| 对比项 | C-PINN (纯 C++) | PyTorch (Python) |
|:-------|:---------------|:----------------|
| **运行时** | 编译型，无解释器 | Python 解释器 + C++ 后端 |
| **部署** | 单个可执行文件（~200K） | Python 环境 + PyTorch 库 |
| **调试** | GDB/LLDB 原生 | Python 栈 + C++ 混合栈 |
| **微分方式** | 有限差分 + 手动反向传播 | Autograd 计算图 |
| **功能丰富度** | 基础 MLP、有限差分 | 完整 DL 生态 |

> ⚠️ 此表仅列出结构差异，未做性能对比。两者定位不同，不适合直接比较速度。

---

## 🗺️ 开发路线图

- [x] **Phase 1**: 张量库（加减乘除、GEMM、切片、广播）
- [x] **Phase 2**: 神经网络前向传播（Fnn, Linear, Activation）
- [x] **Phase 3**: 手动反向传播（梯度计算、链式法则）
- [x] **Phase 4**: 优化器（Adam、L-BFGS）
- [x] **Phase 5**: 模型持久化（二进制 Checkpoint）
- [x] **Phase 6**: PINN 求解验证（KdV, Sine-Gordon, Allen-Cahn）
- [x] **Phase 7A**: 轻量级自动微分原型（标量嵌套图 + `create_graph=true` 机制）
- [x] **Phase 7B**: 矩阵前向原型（含 `matmul`）+ mini PINN 机制验证
- [ ] **Phase 7C**: 与主训练路径集成（`src/nn` + PDE residual stack）
- [ ] **Phase 8**: 更多架构（ResNet, U-Net, FNO）
- [ ] **Phase 9**: GPU 加速（CUDA Kernel / Vulkan Compute）
- [ ] **Phase 10**: Python 绑定（pybind11）

---

## 📖 相关文档

- [docs/pure_c_vs_torch.md](docs/pure_c_vs_torch.md) - 详细技术对比
- [docs/requirements.md](docs/requirements.md) - 需求文档
- [benchmark/autodiff_nested_graph_tech_report.md](benchmark/autodiff_nested_graph_tech_report.md) - 嵌套图 AD 技术报告
- [benchmark/autodiff_nested_graph_test_report.md](benchmark/autodiff_nested_graph_test_report.md) - 标量 AD 细致测试报告
- [benchmark/autodiff_matrix_graph_test_report.md](benchmark/autodiff_matrix_graph_test_report.md) - 矩阵 AD 与回归报告
- [AGENTS.md](AGENTS.md) - 当前开发与验证工作流摘要
- [CLAUDE.md](CLAUDE.md) - AI 开发指南

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

## 📜 许可证

本项目采用 **MIT License** 开源，详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 设计灵感来自 [DeepXDE](https://github.com/lululxvi/deepxde)
- 优化器实现参考 [Adam 论文](https://arxiv.org/abs/1412.6980)
- 感谢 [nlohmann/json](https://github.com/nlohmann/json) 提供的 JSON 库

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/your-username/c-pinn/issues)
- 发起 [Discussion](https://github.com/your-username/c-pinn/discussions)

---

<div align="center">

**Star ⭐ 本项目** 如果你觉得有帮助！

Made with ❤️ using C++17

</div>
