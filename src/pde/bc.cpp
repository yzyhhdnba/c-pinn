#include "pinn/pde/boundary_condition.hpp"

#include <torch/torch.h>

namespace pinn::pde {

DirichletBC::DirichletBC(const geometry::Geometry& geom, ValueFunction value_fn)
    : BoundaryCondition(geom), value_fn_{std::move(value_fn)} {}

Tensor DirichletBC::loss(nn::Fnn& network, const Tensor& points, const Tensor& predicted) const {
    (void)network;
    auto target = value_fn_(points);
    return torch::mse_loss(predicted, target, torch::Reduction::Mean);
}

NeumannBC::NeumannBC(const geometry::Geometry& geom, FluxFunction flux_fn)
    : BoundaryCondition(geom), flux_fn_{std::move(flux_fn)} {}

Tensor NeumannBC::loss(nn::Fnn& network, const Tensor& points, const Tensor& predicted) const {
    (void)network;
    auto target = flux_fn_(points);
    
    // Compute normal derivative: d(u)/dn = grad(u) . n
    auto grad_outputs = torch::ones_like(predicted);
    auto grads = torch::autograd::grad({predicted}, {points}, {grad_outputs}, true, true)[0];
    auto normals = geometry_.boundary_normal(points);
    
    // Dot product along dimension 1
    auto normal_derivs = (grads * normals).sum(1, true);
    
    return torch::mse_loss(normal_derivs, target, torch::Reduction::Mean);
}

PeriodicBC::PeriodicBC(const geometry::Geometry& geom, MappingFunction mapping_fn, Scalar weight)
    : BoundaryCondition(geom), mapping_fn_{std::move(mapping_fn)}, weight_{weight} {}

Tensor PeriodicBC::loss(nn::Fnn& network, const Tensor& points, const Tensor& predicted) const {
    auto mapped_points = mapping_fn_(points);
    auto mapped_predictions = network->forward(mapped_points);
    auto periodic_loss = torch::mse_loss(predicted, mapped_predictions, torch::Reduction::Mean);
    if (weight_ == 1.0) {
        return periodic_loss;
    }
    return periodic_loss * weight_;
}

}  // namespace pinn::pde
