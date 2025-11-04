#pragma once

#include <memory>
#include <vector>

#include "pinn/loss/loss_terms.hpp"
#include "pinn/nn/fnn.hpp"
#include "pinn/pde/boundary_condition.hpp"
#include "pinn/pde/pde.hpp"

namespace pinn::model {

struct LossWeights {
    Scalar pde{1.0};
    Scalar boundary{1.0};
    Scalar data{0.0};
};

class Model {
  public:
    Model(std::shared_ptr<nn::Fnn> network,
          std::shared_ptr<pde::Pde> pde,
          std::vector<std::shared_ptr<pde::BoundaryCondition>> boundary_conditions,
          LossWeights weights = {});

    nn::Fnn& network() { return *network_; }
    const nn::Fnn& network() const { return *network_; }

    pde::Pde& pde() { return *pde_; }
    const pde::Pde& pde() const { return *pde_; }

    const std::vector<std::shared_ptr<pde::BoundaryCondition>>& boundary_conditions() const { return boundary_conditions_; }

    LossWeights& loss_weights() { return weights_; }
    const LossWeights& loss_weights() const { return weights_; }

  private:
    std::shared_ptr<nn::Fnn> network_;
    std::shared_ptr<pde::Pde> pde_;
    std::vector<std::shared_ptr<pde::BoundaryCondition>> boundary_conditions_;
    LossWeights weights_;
};

}  // namespace pinn::model
