# C++ 物理信息神经网络框架

本项目使用现代 C++17 配合 LibTorch 自动微分与优化，复刻 DeepXDE 的核心 PINN 功能。

## 模块概览

- `geometry`：几何域表示、采样策略、CSG 操作。
- `pde`：偏微分方程残差定义、边界/初始条件、运行时 PDE 解析器。
- `nn`：前馈/残差网络结构、激活与初始化、LibTorch 集成。
- `loss`：残差、边界、数据损失计算与权重管理。
- `model`：模型拼装、训练循环、优化器调度、断点保存。
- `utils`：配置解析、回调、指标、日志。

## 快速上手

### 环境依赖

- 支持 C++17 的编译器（建议 GCC 11+/Clang 14+）
- CMake ≥ 3.22
- [LibTorch](https://pytorch.org/get-started/locally/)（与编译器/CUDA 版本匹配）
- [nlohmann_json](https://github.com/nlohmann/json)（CMake 包 `nlohmann_json::nlohmann_json`）
- [Eigen3](https://eigen.tuxfamily.org/)（CMake 包 `Eigen3::Eigen`）

先导出 `CMAKE_PREFIX_PATH` 以便 CMake 找到 LibTorch（以及其它非系统路径的依赖）：

```bash
export CMAKE_PREFIX_PATH="/opt/libtorch"
export Torch_DIR="$CMAKE_PREFIX_PATH/share/cmake/Torch"
```

若尚未安装 LibTorch，可下载与当前 CUDA 运行时匹配的预编译包（或 CPU 版），解压后再设置 `CMAKE_PREFIX_PATH`。要使用 GPU，请选择与驱动兼容的 `+cuXXX` 版本（例如 CUDA 12.8）：

```bash
cd $HOME
wget https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.9.0%2Bcu128.zip
unzip libtorch-shared-with-deps-2.9.0+cu128.zip
export CMAKE_PREFIX_PATH="$HOME/libtorch"
export Torch_DIR="$CMAKE_PREFIX_PATH/share/cmake/Torch"
```

若系统未装 `cmake`（且具备 sudo 权限）：

```bash
sudo apt update
sudo apt install cmake
```

没有 sudo？可在 Conda 环境中安装：

```bash
conda install -c conda-forge cmake
```

同理，JSON/Eigen 等头文件依赖也可通过 Conda 安装：

```bash
conda install -c conda-forge nlohmann_json eigen
```

或下载现成的预编译套件，将其 `bin` 目录加入 `PATH`。

### 编译项目

```bash
mkdir -p build
cd build
TORCH_CUDA_ARCH_LIST="12.0" cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel
```

> 注：若使用 CPU 版或其它架构，可根据需要调整 `TORCH_CUDA_ARCH_LIST` 或直接省略。

### 运行示例/启动项目

1. 构建完成后直接运行内置的 Poisson PINN 演示：

	```bash
	./examples/example_poisson
	```

	程序会自动检测 GPU：若当前 LibTorch 为 CUDA 版且显卡可用，将把模型与数据迁移到 GPU；否则自动回退到 CPU。首次运行需等待数秒完成首个 epoch。

2. 如需调整网络规模、采样数量或训练轮数，可编辑 `config/pinn_config.json`，其中默认配置已匹配 GPU 友好的轻量级设置：

	```json
	{
	  "model": { "layers": [16, 16, 16] },
	  "training": { "batch_size": 96, "epochs": 5 },
	  "data": { "n_interior": 96, "n_boundary": 24 }
	}
	```

	后续代码将把该配置与训练器串联，实现完全配置化启动。

3. 若要自行扩展任务，可以：
	- 在 `examples/` 目录中添加新的方程案例；
	- 扩充 `config/` 以覆盖更多训练场景；
	- 在 `tests/` 中写入单元测试，验证几何采样、损失计算等模块。

完成训练后，`./ckpt`（默认）将按配置保存断点文件，方便继续训练或评估。
