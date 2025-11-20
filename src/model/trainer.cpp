#include "pinn/model/trainer.hpp"

#include <algorithm>
#include <tuple>
#include <unordered_set>

#include <torch/nn/utils/clip_grad.h>
#include <torch/torch.h>

#include "pinn/geometry/sampling.hpp"
#include "pinn/loss/loss_terms.hpp"
#include "pinn/utils/callback.hpp"
#include "pinn/utils/logger.hpp"

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

Tensor Trainer::select_rar_points(torch::Generator& generator, const torch::Device& device) {
    const auto& rar = options_.rar;
    if (!rar.enabled || rar.candidate_pool <= 0 || rar.select_count <= 0) {
        return {};
    }

    auto& domain = model_.pde().domain();
    auto candidates = geometry::sample_interior(domain, rar.candidate_pool, rar.sampling_strategy, generator);
    if (!candidates.defined() || candidates.numel() == 0) {
        return {};
    }

    auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat).device(device);
    candidates = candidates.to(tensor_opts);

    auto residuals = loss::compute_pde_residual(model_.pde(), model_.network(), candidates);
    auto residual_norm = torch::abs(residuals).detach();
    if (residual_norm.dim() > 1) {
        residual_norm = residual_norm.norm(2, 1);
    } else {
        residual_norm = residual_norm.reshape({-1});
    }
    if (residual_norm.numel() == 0) {
        return {};
    }

    const int select = std::min(rar.select_count, static_cast<int>(residual_norm.size(0)));
    if (select <= 0) {
        return {};
    }

    auto topk = residual_norm.topk(select, 0, true, true);
    auto indices = std::get<1>(topk);
    return candidates.index_select(0, indices);
}

void Trainer::maybe_apply_rar(TrainingBatch& batch,
                              int epoch,
                              torch::Generator& generator,
                              const torch::Device& device) {
    const auto& rar = options_.rar;
    if (!rar.enabled || rar.candidate_pool <= 0 || rar.select_count <= 0) {
        return;
    }

    const int frequency = (rar.apply_every <= 0) ? 1 : rar.apply_every;
    if (epoch % frequency == 0) {
        auto rar_points = select_rar_points(generator, device);
        if (rar_points.defined() && rar_points.numel() > 0) {
            if (rar_buffer_.defined() && rar_buffer_.numel() > 0) {
                rar_buffer_ = torch::cat({rar_buffer_, rar_points}, 0);
            } else {
                rar_buffer_ = rar_points;
            }
        }
    }

    if (rar_buffer_.defined() && rar_buffer_.numel() > 0) {
        if (batch.interior_points.defined() && batch.interior_points.numel() > 0) {
            batch.interior_points = torch::cat({batch.interior_points, rar_buffer_}, 0);
        } else {
            batch.interior_points = rar_buffer_;
        }
    }
}

void Trainer::fit(const TrainingBatch& batch, utils::CallbackRegistry& callbacks) {
    configure_optimizers();

    std::unordered_set<int> milestones(options_.schedule.milestones.begin(), options_.schedule.milestones.end());
    
    const int effective_mini_batch = (options_.mini_batch_size > 0) ? options_.mini_batch_size : options_.batch_size;
    const int n_interior = batch.interior_points.size(0);
    const int n_boundary = batch.boundary_points.defined() ? batch.boundary_points.size(0) : 0;

    for (int epoch = 0; epoch < options_.epochs; ++epoch) {
        utils::CallbackContext ctx;
        ctx.epoch = epoch;

        callbacks.epoch_begin(ctx);

        // Mini-batch 训练循环
        const int num_mini_batches = std::max(1, (n_interior + effective_mini_batch - 1) / effective_mini_batch);
        double epoch_loss = 0.0;
        
        for (int mb = 0; mb < num_mini_batches; ++mb) {
            optimizer_->zero_grad();
            
            // 切片当前 mini-batch
            const int start = mb * effective_mini_batch;
            const int end = std::min(start + effective_mini_batch, n_interior);
            
            auto mb_interior = batch.interior_points.slice(0, start, end);
            
            // 边界点按比例采样（简化处理：均匀分配）
            Tensor mb_boundary;
            Tensor mb_boundary_values;
            if (n_boundary > 0) {
                const int b_start = (mb * n_boundary) / num_mini_batches;
                const int b_end = ((mb + 1) * n_boundary) / num_mini_batches;
                mb_boundary = batch.boundary_points.slice(0, b_start, b_end);
                if (batch.boundary_values.defined()) {
                    mb_boundary_values = batch.boundary_values.slice(0, b_start, b_end);
                }
            }

            auto losses = loss::compute_losses(model_.pde(),
                                               model_.boundary_conditions(),
                                               mb_interior,
                                               mb_boundary,
                                               mb_boundary_values,
                                               model_.network(),
                                               model_.loss_weights().pde,
                                               model_.loss_weights().boundary,
                                               model_.loss_weights().data);

            losses.total_loss.backward();

            if (options_.schedule.gradient_clip_norm > 0.0) {
                torch::nn::utils::clip_grad_norm_(model_.network()->parameters(), options_.schedule.gradient_clip_norm);
            }

            optimizer_->step();
            
            epoch_loss += losses.total_loss.item<double>();
        }

        ctx.loss = epoch_loss / num_mini_batches;
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

void Trainer::fit_with_resampling(SamplingFunction sampler,
                                  torch::Generator& generator,
                                  const torch::Device& device,
                                  utils::CallbackRegistry& callbacks) {
    configure_optimizers();

    std::unordered_set<int> milestones(options_.schedule.milestones.begin(), options_.schedule.milestones.end());
    const int effective_mini_batch = (options_.mini_batch_size > 0) ? options_.mini_batch_size : options_.batch_size;

    for (int epoch = 0; epoch < options_.epochs; ++epoch) {
        utils::CallbackContext ctx;
        ctx.epoch = epoch;

        callbacks.epoch_begin(ctx);

        // 每个 epoch 重新采样
        auto batch = sampler(device, generator);
        maybe_apply_rar(batch, epoch, generator, device);
        
        const int n_interior = batch.interior_points.size(0);
        const int n_boundary = batch.boundary_points.defined() ? batch.boundary_points.size(0) : 0;
        const int num_mini_batches = std::max(1, (n_interior + effective_mini_batch - 1) / effective_mini_batch);
        
        double epoch_loss = 0.0;
        
        for (int mb = 0; mb < num_mini_batches; ++mb) {
            optimizer_->zero_grad();
            
            const int start = mb * effective_mini_batch;
            const int end = std::min(start + effective_mini_batch, n_interior);
            
            auto mb_interior = batch.interior_points.slice(0, start, end);
            
            Tensor mb_boundary;
            Tensor mb_boundary_values;
            if (n_boundary > 0) {
                const int b_start = (mb * n_boundary) / num_mini_batches;
                const int b_end = ((mb + 1) * n_boundary) / num_mini_batches;
                mb_boundary = batch.boundary_points.slice(0, b_start, b_end);
                if (batch.boundary_values.defined()) {
                    mb_boundary_values = batch.boundary_values.slice(0, b_start, b_end);
                }
            }

            auto losses = loss::compute_losses(model_.pde(),
                                               model_.boundary_conditions(),
                                               mb_interior,
                                               mb_boundary,
                                               mb_boundary_values,
                                               model_.network(),
                                               model_.loss_weights().pde,
                                               model_.loss_weights().boundary,
                                               model_.loss_weights().data);

            losses.total_loss.backward();

            if (options_.schedule.gradient_clip_norm > 0.0) {
                torch::nn::utils::clip_grad_norm_(model_.network()->parameters(), options_.schedule.gradient_clip_norm);
            }

            optimizer_->step();
            epoch_loss += losses.total_loss.item<double>();
        }

        ctx.loss = epoch_loss / num_mini_batches;
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
