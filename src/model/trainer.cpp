#include "pinn/model/trainer.hpp"

#include <algorithm>
#include <unordered_set>

#include <torch/nn/utils/clip_grad.h>
#include <torch/torch.h>

#include "pinn/loss/loss_terms.hpp"
#include "pinn/utils/callback.hpp"

namespace pinn::model {

Trainer::Trainer(Model& model, TrainingOptions options)
    : model_{model}, options_{std::move(options)} {}

void Trainer::configure_optimizers() {
    if (optimizers_ready_) {
        return;
    }
    auto adam_options = torch::optim::AdamOptions(options_.schedule.learning_rate);
    optimizer_ = std::make_unique<torch::optim::Adam>(model_.network()->parameters(), adam_options);

    if (options_.schedule.switch_to_lbfgs_epoch >= 0) {
        auto lbfgs_options = torch::optim::LBFGSOptions(options_.schedule.learning_rate);
        lbfgs_optimizer_ = std::make_unique<torch::optim::LBFGS>(model_.network()->parameters(), lbfgs_options);
    }

    optimizers_ready_ = true;
}

void Trainer::fit(const TrainingBatch& batch, utils::CallbackRegistry& callbacks) {
    configure_optimizers();

    std::unordered_set<int> milestones(options_.schedule.milestones.begin(), options_.schedule.milestones.end());

    for (int epoch = 0; epoch < options_.epochs; ++epoch) {
        utils::CallbackContext ctx;
        ctx.epoch = epoch;

        callbacks.epoch_begin(ctx);

        optimizer_->zero_grad();

        auto losses = loss::compute_losses(model_.pde(),
                                           model_.boundary_conditions(),
                                           batch.interior_points,
                                           batch.boundary_points,
                                           batch.boundary_values,
                                           model_.network(),
                                           model_.loss_weights().pde,
                                           model_.loss_weights().boundary,
                                           model_.loss_weights().data);

        losses.total_loss.backward();

        if (options_.schedule.gradient_clip_norm > 0.0) {
            torch::nn::utils::clip_grad_norm_(model_.network()->parameters(), options_.schedule.gradient_clip_norm);
        }

        optimizer_->step();

        ctx.loss = losses.total_loss.item<double>();
        callbacks.epoch_end(ctx);

        if (milestones.count(epoch) > 0) {
            auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer_.get());
            if (adam != nullptr) {
                for (auto& group : adam->param_groups()) {
                    auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
                    options.lr(options.lr() * options_.schedule.gamma);
                }
            }
        }

        if (options_.schedule.switch_to_lbfgs_epoch >= 0 && epoch == options_.schedule.switch_to_lbfgs_epoch) {
            // TODO: Implement LBFGS phase using closure-based optimization.
        }
    }
}

}  // namespace pinn::model
