# C++ 物理信息神经网络 (PINN) 框架

本项目是一个基于现代 C++17 和 LibTorch 的高性能物理信息神经网络（Physics-Informed Neural Networks, PINN）框架。它旨在复刻并扩展 DeepXDE 的核心功能，提供从几何建模、PDE 定义到模型训练与可视化的全套解决方案。

## ✨ 核心特性

- **多后端支持**：基于 LibTorch，支持 CPU 和 CUDA 加速（自动检测）。
- **丰富的网络架构**：
  - **FNN** (全连接网络)
  - **ResNet** / **ModifiedResNet** (残差网络)
  - **CNN** (卷积神经网络，支持 1D/2D/3D)
  - **Transformer** (自注意力机制)
- **先进的激活函数**：
  - 基础：`tanh`, `sigmoid`, `relu`, `leaky_relu`, `gelu`, `swish`, `silu`
  - 自适应：`adaptive_tanh`, `adaptive_sigmoid`, `rowdy`, `l_lcaf` (Layer-wise Locally Adaptive Activation Functions)
- **高级训练策略**：
  - **RAR** (Residual-based Adaptive Refinement)：基于残差的自适应加点策略，自动提升难点区域的精度。
  - **L-BFGS**：支持二阶优化器微调。
  - **学习率调度**：StepLR, ExponentialLR 等。
- **边界条件**：
  - Dirichlet, Neumann, Robin
  - **PeriodicBC** (周期性边界条件)
- **几何与采样**：
  - 支持区间、矩形等基础几何。
  - 支持 Latin Hypercube Sampling (LHS)、Grid、Random 等采样策略。

## 🚀 快速上手

### 1. 环境准备

请确保您的系统已安装以下依赖：

- **C++ 编译器**：支持 C++17 (GCC >= 9, Clang >= 11, MSVC >= 19.28)
- **CMake**：>= 3.15
- **LibTorch**：C++ 版 PyTorch (建议 2.0+)
- **Eigen3**：线性代数库
- **nlohmann_json**：JSON 解析库
- **OpenMP** (可选，推荐)：用于并行加速

#### macOS (Apple Silicon)

```bash
# 安装基础依赖
brew install cmake eigen nlohmann-json libomp

# 下载 LibTorch (CPU 版)
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.5.1.zip
unzip libtorch-shared-with-deps-2.5.1.zip
export LIBTORCH_PATH=$(pwd)/libtorch
```

#### Linux (Ubuntu)

```bash
sudo apt-get install cmake libeigen3-dev nlohmann-json3-dev

# 下载 LibTorch (CUDA 版，请根据显卡选择版本)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu118.zip
export LIBTORCH_PATH=$(pwd)/libtorch
```

### 2. 编译项目

```bash
mkdir build
cd build

# 配置 CMake (需指定 LibTorch 路径)
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ..

# 编译 (使用多核)
cmake --build . --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### 3. 运行示例

编译完成后，可执行文件位于 `build/examples/` 目录。

**Burgers 方程示例**：

```bash
# 运行 Burgers 方程求解
./build/examples/example_burgers config/burgers_config.json
```

程序将在 `sandbox/burgers/` 目录下生成训练过程的 CSV 文件（包含预测值、真实值和误差）。

## ⚙️ 配置说明

项目使用 JSON 文件进行配置，支持覆盖默认参数。

### 配置文件结构 (`config/burgers_config.json` 示例)

```json
{
  "model": {
    "input_dim": 2,
    "output_dim": 1,
    "layers": [64, 64, 64, 64],
    "activation": "tanh",          // 支持 "adaptive_tanh", "swish" 等
    "architecture": "fnn",         // 可选: "fnn", "resnet", "modified_resnet", "transformer"
    "weight_init": "xavier_uniform"
  },
  "training": {
    "optimizer": "adam",
    "lr": 0.001,
    "epochs": 1000,
    "rar_enabled": true,           // 启用 RAR 自适应加点
    "rar_frequency": 100,          // 每 100 epoch 加一次点
    "rar_topk": 100                // 每次添加残差最大的 100 个点
  },
  "data": {
    "n_interior": 2000,
    "n_boundary": 500,
    "sampling": "latin_hypercube"
  },
  "pde": {
    "nu": 0.01                     // PDE 特定参数
  }
}
```

### 命令行参数

配置文件的加载优先级：
1. 命令行参数：`./exe path/to/config.json`
2. 环境变量：`PINN_CONFIG=path/to/config.json ./exe`
3. 默认路径：`config/<name>_config.json`

## 📊 可视化

训练过程中生成的 CSV 文件可以使用 Python 脚本进行绘图：

```bash
# 安装绘图依赖
pip install matplotlib pandas numpy

# 绘制结果
python scripts/plot_csv.py sandbox/burgers/burgers_epoch_00100.csv --output result.png
```

## 📂 目录结构

```
.
├── CMakeLists.txt          # 构建脚本
├── config/                 # 配置文件
├── examples/               # 示例代码 (Burgers, Poisson, Advection)
├── include/pinn/           # 头文件
│   ├── geometry/           # 几何定义
│   ├── loss/               # 损失函数
│   ├── model/              # 模型与训练器
│   ├── nn/                 # 神经网络架构
│   └── pde/                # PDE 定义与边界条件
├── src/                    # 源代码
└── scripts/                # 辅助脚本
```

## 🛠️ 常见问题

1. **`dyld: Library not loaded` (macOS)**:
   - 确保 `libomp` 已安装 (`brew install libomp`)。
   - 确保 LibTorch 的库路径已添加到 `DYLD_LIBRARY_PATH` (虽然 CMake RPATH 通常会处理好)。

2. **内存不足 (OOM)**:
   - 减小 `batch_size` 或 `n_interior`。
   - 减小网络规模 (`layers`)。
   - 在 macOS 上，注意 LibTorch CPU 版的内存占用。

3. **CMake 找不到 Torch**:
   - 确保 `CMAKE_PREFIX_PATH` 正确指向解压后的 `libtorch` 目录。


## 快速上手

### 一分钟总览

1. 安装依赖：LibTorch、CMake、nlohmann_json、Eigen3，以及支持 C++17 的编译器。
2. 进入仓库根目录执行 `mkdir -p build && cd build && TORCH_CUDA_ARCH_LIST="12.0" cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . --parallel`。
3. 回到仓库根目录，运行 `./build/examples/example_poisson`，观察 `sandbox/poisson/` 下是否产生 CSV。

### 依赖准备

- 支持 C++17 的编译器（建议 GCC ≥ 11、Clang ≥ 14）
- CMake ≥ 3.22
- [LibTorch](https://pytorch.org/get-started/locally/)（版本需与编译器和 CUDA 兼容）
- [nlohmann_json](https://github.com/nlohmann/json)
- [Eigen3](https://eigen.tuxfamily.org/)

推荐流程：

```bash
# 1. 从 PyTorch 官网下载与 CUDA 版本匹配的 LibTorch 包
LIBTORCH_ZIP=libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip   # 示例，具体版本请以官网为准
wget https://download.pytorch.org/libtorch/cu121/$LIBTORCH_ZIP
unzip $LIBTORCH_ZIP -d $HOME

# 2. 告诉 CMake LibTorch 的安装路径
export CMAKE_PREFIX_PATH="$HOME/libtorch"
export Torch_DIR="$CMAKE_PREFIX_PATH/share/cmake/Torch"

# 3a. Conda 环境：
conda install -c conda-forge cmake nlohmann_json eigen

# 3b. 或使用 APT（具备 sudo 权限时）：
sudo apt update
sudo apt install cmake nlohmann-json3-dev libeigen3-dev
```

若使用 CPU 版 LibTorch，请从 CPU 下载页面获取 `libtorch-shared-with-deps-*-cpu.zip`，并省略 CUDA 相关变量。

### 构建

```bash
mkdir -p build
cd build
TORCH_CUDA_ARCH_LIST="12.0" cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
```

- `TORCH_CUDA_ARCH_LIST` 需按目标 GPU 的计算能力设置（例如 8.6、9.0）；使用 CPU 版 LibTorch 时可删除该变量。
- 构建完成后，示例二进制位于 `build/examples/`。

### 运行流程

1. 进入仓库根目录运行示例：

    ```bash
    cd /home/yzy/work4
    ./build/examples/example_poisson
    ```

  建议始终在仓库根目录启动，这样 CSV 会落在 `sandbox/<示例名>/`，与脚本默认路径一致。

2. 执行时若 `torch::cuda::is_available()` 返回 true，程序会使用 `torch::Device(torch::kCUDA, 0)`；否则自动回退到 CPU 并输出提示。

3. 配置优先级：命令行参数 > 环境变量 `PINN_CONFIG` > 默认 JSON。配置文件位于 `config/`，可直接拷贝后修改。

4. 训练默认注册两个回调：一个打印损失，一个按照约总 epoch 数 10% 的频率导出可视化 CSV。可在示例源码中调整 `VisualizationOptions` 改变频率与路径。

### 示例详解

#### Poisson（一维方程）

- 启动命令：

  ```bash
  ./build/examples/example_poisson [可选配置路径]
  ```

- 默认配置：`config/pinn_config.json`
- PDE：$u''(x) + \pi^2 \sin(\pi x) = 0$，边界条件 $u(0)=u(1)=0$
- 数据采样：区间 $[0,1]$ 上均匀 256 个评估点；训练点数由配置决定
- 可视化输出：`sandbox/poisson/poisson_epoch_XXXXX.csv`
- 典型配置片段：

  ```json
  {
    "model": { "layers": [1, 16, 16, 16, 1] },
    "training": { "epochs": 5, "batch_size": 96 },
    "data": { "n_interior": 96, "n_boundary": 24 }
  }
  ```

#### Advection（一维空间 + 时间）

- 启动命令：

  ```bash
  ./build/examples/example_advection [可选配置路径]
  ```

- 默认配置：`config/advection_config.json`
- PDE：$u_t + c u_x = 0$，解析解 $u(x,t)=\sin(\pi[x-ct])$
- 特性：矩形区域 $(x,t)\in[0,1]^2$，配置文件中 `pde.velocity` 控制速度 $c$
- 可视化输出：`sandbox/advection/advection_epoch_XXXXX.csv`，包含网格化的 `(x,t)` 点及预测/解析值
- 默认采样：64×64 规则网格用于可视化

#### Burgers（粘性方程）

- 启动命令：

  ```bash
  ./build/examples/example_burgers [可选配置路径]
  ```

- 默认配置：`config/burgers_config.json`
- PDE：$u_t + u u_x - \nu u_{xx} = f(x,t)$，默认粘性系数 `nu = 0.01`
- 特性：同样在 $(x,t)\in[0,1]^2$ 上训练，强制项和边界条件按照解析解构造
- 可视化输出：`sandbox/burgers/burgers_epoch_XXXXX.csv`
- 默认可视化网格：64×64

### 配置与扩展

- 命令行覆盖：运行示例时附带自定义配置路径，例如 `./build/examples/example_burgers ../my_configs/burgers.json`
- 环境变量覆盖：`PINN_CONFIG=/path/to/custom.json ./build/examples/example_advection`
- 内置 JSON 字段说明：
  - `model.layers`：网络层宽（含输入/输出）
  - `training.batch_size | epochs | learning_rate | use_lbfgs_after` 等训练参数
  - `data.n_interior | n_boundary`：每个 epoch 的采样数量
  - `pde.*`：各示例特有的 PDE 参数，如 advection 的 `velocity`、burgers 的 `nu`
- 自定义示例：复制现有 `examples/*.cpp`，在 `CMakeLists.txt` 的 `add_subdirectory(examples)` 中注册新目标即可。

### GPU 选择

- 使用环境变量限制可见设备：

  ```bash
  CUDA_VISIBLE_DEVICES=2 ./build/examples/example_poisson
  ```

- 程序内部永远创建 `torch::Device(torch::kCUDA, 0)`，因此上述命令会把宿主机的 GPU 2 暴露为进程内的 CUDA 设备 0。
- 需要切换不同 GPU 时，直接调整 `CUDA_VISIBLE_DEVICES` 的值；未设置或没有可见 GPU 时自动回退到 CPU。

### 可视化导出与绘图

- CSV 位置：相对于程序启动时的工作目录创建 `sandbox/<示例名>/`；若从 `build/` 目录运行，文件会位于 `build/sandbox/...`。建议总是在仓库根目录运行，使路径与脚本默认值一致。
- CSV 结构：
  - 坐标列：`x0,x1,...`
  - 模型输出：`pred0,...`
  - 若提供解析解：`target0,...`
  - 绝对误差：`abs_error0,...`
- 绘图脚本：

  ```bash
  pip install matplotlib pandas numpy
  python scripts/plot_csv.py sandbox/poisson/poisson_epoch_00004.csv --output poisson.png
  python scripts/plot_csv.py sandbox/advection/advection_epoch_00004.csv --dim 2 --output advection.png
  ```

- 若 CSV 位于 `build/sandbox/...`，请在命令行中按实际路径替换。
- 生成图片后可用于对比预测与解析解，也能用于误差热力图等二次分析。

### 训练输出与断点

- 训练日志：每个 epoch 结束后打印损失值，便于跟踪收敛情况。
- 断点：默认写入 `ckpt/` 目录，可在配置文件里调整保存策略或频率。
- 扩展建议：
  - 需要更多监控指标时，可在 `utils::CallbackRegistry` 中注册新的回调。
  - 若要启用 LBFGS 二阶段优化，可在 `TrainingOptions.schedule.switch_to_lbfgs_epoch` 设置切换 epoch，并在 `Trainer` 中补充相应实现。

