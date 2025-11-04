#include "pinn/pde/boundary_condition.hpp"

namespace pinn::pde {

DirichletBC::DirichletBC(const geometry::Geometry& geom, ValueFunction value_fn)
    : BoundaryCondition(geom), value_fn_{std::move(value_fn)} {}

Tensor DirichletBC::loss(const Tensor& points, const Tensor& predicted) const {
    auto target = value_fn_(points);
    return torch::mse_loss(predicted, target, torch::Reduction::Mean);
}

NeumannBC::NeumannBC(const geometry::Geometry& geom, FluxFunction flux_fn)
    : BoundaryCondition(geom), flux_fn_{std::move(flux_fn)} {}

Tensor NeumannBC::loss(const Tensor& points, const Tensor& predicted) const {
    auto target = flux_fn_(points);
    return torch::mse_loss(predicted, target, torch::Reduction::Mean);
}

}  // namespace pinn::pde
