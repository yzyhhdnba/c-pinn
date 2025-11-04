#include "pinn/model/model.hpp"

namespace pinn::model {

Model::Model(std::shared_ptr<nn::Fnn> network,
             std::shared_ptr<pde::Pde> pde,
             std::vector<std::shared_ptr<pde::BoundaryCondition>> boundary_conditions,
             LossWeights weights)
    : network_{std::move(network)},
      pde_{std::move(pde)},
      boundary_conditions_{std::move(boundary_conditions)},
      weights_{weights} {}

}  // namespace pinn::model
