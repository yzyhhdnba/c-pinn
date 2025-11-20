#include "pinn/loss/loss_terms.hpp"

#include <torch/torch.h>

#include "pinn/utils/logger.hpp"

namespace pinn::loss {

namespace {
Tensor compute_gradients(const Tensor& outputs, Tensor inputs) {
    const auto batch = outputs.size(0);
    const auto dim = inputs.size(1);
    auto gradients = torch::zeros({batch, dim}, inputs.options());

    for (int64_t i = 0; i < batch; ++i) {
        auto grad_outputs = torch::zeros_like(outputs);
        grad_outputs[i] = 1.0;
        auto grad = torch::autograd::grad({outputs}, {inputs}, {grad_outputs}, true, true)[0];
        gradients.select(0, i).copy_(grad.select(0, i));
    }
    return gradients;
}

Tensor compute_hessian(const Tensor& outputs, Tensor inputs) {
    const auto batch = outputs.size(0);
    const auto dim = inputs.size(1);
    auto hessian = torch::zeros({batch, dim, dim}, inputs.options());

    auto gradients = compute_gradients(outputs, inputs);
    for (int64_t axis = 0; axis < dim; ++axis) {
        auto component = gradients.select(1, axis);
        for (int64_t i = 0; i < batch; ++i) {
            auto grad_outputs = torch::zeros_like(component);
            grad_outputs[i] = 1.0;
            auto second = torch::autograd::grad({component}, {inputs}, {grad_outputs}, true, true)[0];
            hessian.select(0, i).select(0, axis).copy_(second.select(0, i));
        }
    }
    return hessian;
}

Tensor compute_boundary_loss(const pde::BoundaryCondition& bc,
                             nn::Fnn& network,
                             const Tensor& points,
                             const Tensor& predictions) {
    return bc.loss(network, points, predictions);
}

}  // namespace

Tensor compute_pde_residual(const pde::Pde& pde,
                            nn::Fnn& network,
                            const Tensor& interior_points) {
    auto inputs = interior_points.clone();
    inputs.set_requires_grad(true);
    auto predictions = network->forward(inputs);
    auto gradients = compute_gradients(predictions, inputs);
    auto hessian = compute_hessian(predictions, inputs);

    pde::DifferentialData data{inputs, predictions, gradients, hessian};
    return pde.evaluate(data);
}

LossBreakdown compute_losses(const pde::Pde& pde,
                             const std::vector<std::shared_ptr<pde::BoundaryCondition>>& boundary_conditions,
                             const Tensor& interior_points,
                             const Tensor& boundary_points,
                             const Tensor& boundary_targets,
                             nn::Fnn& network,
                             Scalar pde_weight,
                             Scalar boundary_weight,
                             Scalar data_weight) {
    LossBreakdown breakdown;

    auto residual = compute_pde_residual(pde, network, interior_points);
    breakdown.pde_loss = torch::mean(torch::pow(residual, 2));

    Tensor boundary_loss = torch::zeros({}, residual.options());
    Tensor boundary_predictions;
    if (!boundary_conditions.empty() && boundary_points.numel() > 0) {
        // Ensure boundary points track gradients for Neumann/Robin BCs
        if (!boundary_points.requires_grad()) {
            boundary_points.set_requires_grad(true);
        }
        boundary_predictions = network->forward(boundary_points);
        for (const auto& bc : boundary_conditions) {
            boundary_loss += compute_boundary_loss(*bc, network, boundary_points, boundary_predictions);
        }
        boundary_loss = boundary_loss / static_cast<double>(boundary_conditions.size());
    }
    breakdown.boundary_loss = boundary_loss;

    if (boundary_targets.defined() && boundary_targets.numel() > 0) {
        if (!boundary_predictions.defined()) {
            boundary_predictions = network->forward(boundary_points);
        }
        breakdown.data_loss = torch::mse_loss(boundary_predictions, boundary_targets, torch::Reduction::Mean);
    } else {
        breakdown.data_loss = torch::zeros({}, residual.options());
    }

    breakdown.total_loss = pde_weight * breakdown.pde_loss + boundary_weight * breakdown.boundary_loss +
                           data_weight * breakdown.data_loss;
    return breakdown;
}

}  // namespace pinn::loss
