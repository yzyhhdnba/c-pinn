#pragma once

#include <string>

#include "pinn/geometry/geometry.hpp"

namespace pinn::geometry {

enum class SamplingStrategy {
    kUniformGrid,
    kLatinHypercube
};

SamplingStrategy sampling_strategy_from_string(const std::string& name);
Tensor sample_interior(const Geometry& geometry, int n_points, SamplingStrategy strategy, pinn::core::Rng& rng);
Tensor sample_boundary(const Geometry& geometry, int n_points, pinn::core::Rng& rng);

}  // namespace pinn::geometry
