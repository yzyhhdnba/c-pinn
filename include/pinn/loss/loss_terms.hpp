#pragma once

#include <memory>
#include <vector>

#include "pinn/pde/boundary_condition.hpp"
#include "pinn/nn/fnn.hpp"
#include "pinn/pde/pde.hpp"
#include "pinn/types.hpp"

namespace pinn::loss {

struct LossBreakdown {
    Tensor pde_loss;
    Tensor boundary_loss;
    Tensor data_loss;
    Tensor total_loss;
};

LossBreakdown compute_losses(const pde::Pde& pde,
                             const std::vector<std::shared_ptr<pde::BoundaryCondition>>& boundary_conditions,
                             const Tensor& interior_points,
                             const Tensor& boundary_points,
                             const Tensor& boundary_targets,
                             nn::Fnn& network,
                             Scalar pde_weight,
                             Scalar boundary_weight,
                             Scalar data_weight);

Tensor compute_pde_residual(const pde::Pde& pde,
                            nn::Fnn& network,
                            const Tensor& interior_points);

}  // namespace pinn::loss
