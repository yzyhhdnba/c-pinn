#pragma once

#include <functional>

#include "pinn/geometry/geometry.hpp"
#include "pinn/types.hpp"

namespace pinn::pde {

struct DifferentialData {
    Tensor point;
    Tensor value;
    Tensor gradients;
    Tensor hessian;
};

using PdeFunction = std::function<Tensor(const DifferentialData&)>;

class Pde {
  public:
    Pde(geometry::Geometry& domain, PdeFunction residual);

    geometry::Geometry& domain() const { return domain_; }
    const PdeFunction& residual() const { return residual_; }

    Tensor evaluate(const DifferentialData& data) const;

  private:
    geometry::Geometry& domain_;
    PdeFunction residual_;
};

}  // namespace pinn::pde
