#pragma once

#include <functional>

#include "pinn/geometry/geometry.hpp"
#include "pinn/types.hpp"

namespace pinn::pde {

class BoundaryCondition {
  public:
    explicit BoundaryCondition(const geometry::Geometry& geom) : geometry_{geom} {}
    virtual ~BoundaryCondition() = default;

    const geometry::Geometry& geometry() const { return geometry_; }

    virtual Tensor loss(const Tensor& points, const Tensor& predicted) const = 0;

  protected:
    const geometry::Geometry& geometry_;
};

class DirichletBC : public BoundaryCondition {
  public:
    using ValueFunction = std::function<Tensor(const Tensor&)>;

    DirichletBC(const geometry::Geometry& geom, ValueFunction value_fn);

    Tensor loss(const Tensor& points, const Tensor& predicted) const override;

  private:
    ValueFunction value_fn_;
};

class NeumannBC : public BoundaryCondition {
  public:
    using FluxFunction = std::function<Tensor(const Tensor&)>;

    NeumannBC(const geometry::Geometry& geom, FluxFunction flux_fn);

    Tensor loss(const Tensor& points, const Tensor& predicted) const override;

  private:
    FluxFunction flux_fn_;
};

}  // namespace pinn::pde
