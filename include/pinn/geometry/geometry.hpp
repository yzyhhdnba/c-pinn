#pragma once

#include <utility>

#include "pinn/types.hpp"

namespace pinn::geometry {

class Geometry {
  public:
    explicit Geometry(int dim) : dimension_{dim} {}
    virtual ~Geometry() = default;

    int dimension() const noexcept { return dimension_; }

    virtual bool inside(const Tensor& x) const = 0;
    virtual bool on_boundary(const Tensor& x) const = 0;
    virtual Tensor boundary_normal(const Tensor& x) const = 0;

    virtual Tensor uniform_points(int n_per_dim) const = 0;
    virtual Tensor random_points(int n, torch::Generator& gen) const = 0;

    virtual std::pair<Tensor, Tensor> bounds() const = 0;

  private:
    int dimension_;
};

}  // namespace pinn::geometry
