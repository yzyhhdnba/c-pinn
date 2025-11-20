#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "pinn/geometry/sampling.hpp"
#include "pinn/model/model.hpp"
#include "pinn/types.hpp"
#include "pinn/utils/callback.hpp"

namespace pinn::model {

struct OptimizerSchedule {
    std::string optimizer_name{"adam"};
    Scalar learning_rate{1e-3};
    std::vector<int> milestones{};
    Scalar gamma{0.1};
    int switch_to_lbfgs_epoch{-1};
    Scalar gradient_clip_norm{1.0};
};

struct RarOptions {
  bool enabled{false};
  int candidate_pool{0};
  int select_count{0};
  int apply_every{10};
  geometry::SamplingStrategy sampling_strategy{geometry::SamplingStrategy::kLatinHypercube};
};

struct TrainingOptions {
    int epochs{1000};
    int batch_size{1024};
    OptimizerSchedule schedule{};
    
    // Mini-batch 训练配置
    int mini_batch_size{0};  // 0 表示使用全部数据，>0 时启用 mini-batch
    
    // 重采样配置（可选）
    bool resample_every_epoch{false};
    int n_interior_points{0};
    int n_boundary_points{0};

    RarOptions rar{};
};

// 采样函数类型定义
using SamplingFunction = std::function<TrainingBatch(const torch::Device&, torch::Generator&)>;

class Trainer {
  public:
    Trainer(Model& model, TrainingOptions options);

    // 原有接口：使用预生成的固定批次
    void fit(const TrainingBatch& batch, utils::CallbackRegistry& callbacks);
    
    // 新接口：支持动态重采样
    void fit_with_resampling(SamplingFunction sampler, 
                            torch::Generator& generator,
                            const torch::Device& device,
                            utils::CallbackRegistry& callbacks);

  private:
    void configure_optimizers();
    Tensor select_rar_points(torch::Generator& generator, const torch::Device& device);
    void maybe_apply_rar(TrainingBatch& batch,
                         int epoch,
                         torch::Generator& generator,
                         const torch::Device& device);

    Model& model_;
    TrainingOptions options_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    std::unique_ptr<torch::optim::Optimizer> lbfgs_optimizer_;
    bool optimizers_ready_{false};
    Tensor rar_buffer_;
};

}  // namespace pinn::model
