# 计算流图目录

本目录存放项目的 Markdown 版计算流图，使用 Mermaid 绘制，内容严格对应当前仓库中的实际实现。

## 文件说明

- [pure_cpp_pinn_flow.md](/Users/hhd/Desktop/test/c-pinn/docs/compute_flows/pure_cpp_pinn_flow.md)  
  纯 C++ 主训练链路：输入采样、FNN 层级、有限差分 stencil、PDE residual、反向传播与参数更新。

- [autodiff_matrix_flow.md](/Users/hhd/Desktop/test/c-pinn/docs/compute_flows/autodiff_matrix_flow.md)  
  `examples/autodiff/matrix_graph.cpp` 中 mini autodiff PINN 原型：`Node` 图、`VarMatrix` 流、`matmul` 层级、PDE 高阶导链路。

## 适用范围

- 纯 C++ 主训练路径：对应 `examples/pure_c_kdv.cpp`、`examples/pure_c_sine_gordon.cpp`、`examples/pure_c_allen_cahn.cpp`
- 最小 autodiff 原型：对应 `examples/autodiff/matrix_graph.cpp`

## 说明

- 图中 shape 使用代码中的真实维度。
- 纯 C++ 主训练图默认以 `batch_size=64`、网络结构 `[2, 50, 50, 50, 1]` 为基准。
- autodiff 原型图默认以 `TinyFNN({2, 4, 1})`、单点输入 `(x, t)` 为基准。
