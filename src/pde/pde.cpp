#include "pinn/pde/pde.hpp"

namespace pinn::pde {

Pde::Pde(geometry::Geometry& domain, PdeFunction residual)
    : domain_{domain}, residual_{std::move(residual)} {}

Tensor Pde::evaluate(const DifferentialData& data) const {
    return residual_(data);
}

}  // namespace pinn::pde
