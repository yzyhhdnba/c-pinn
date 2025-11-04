#pragma once

#include <memory>
#include <vector>

#include <torch/torch.h>

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

struct TrainingOptions {
    int epochs{1000};
    int batch_size{1024};
    OptimizerSchedule schedule{};
};

class Trainer {
  public:
    Trainer(Model& model, TrainingOptions options);

    void fit(const TrainingBatch& batch, utils::CallbackRegistry& callbacks);

  private:
    void configure_optimizers();

    Model& model_;
    TrainingOptions options_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    std::unique_ptr<torch::optim::Optimizer> lbfgs_optimizer_;
    bool optimizers_ready_{false};
};

}  // namespace pinn::model
