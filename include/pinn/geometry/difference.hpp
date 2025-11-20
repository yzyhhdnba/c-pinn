#pragma once

#include <memory>

#include "pinn/geometry/geometry.hpp"

namespace pinn::geometry {

class Difference : public Geometry {
  public:
    Difference(std::shared_ptr<Geometry> primary,
               std::shared_ptr<Geometry> subtract);

    bool inside(const Tensor& x) const override;
    bool on_boundary(const Tensor& x) const override;
    Tensor boundary_normal(const Tensor& x) const override;

    Tensor uniform_points(int n_per_dim) const override;
    Tensor random_points(int n, torch::Generator& gen) const override;

    std::pair<Tensor, Tensor> bounds() const override;

    const Geometry& primary() const { return *primary_; }
    const Geometry& subtract() const { return *subtract_; }

  private:
    Tensor filter_valid(const Tensor& candidates) const;

    std::shared_ptr<Geometry> primary_;
    std::shared_ptr<Geometry> subtract_;
};

}  // namespace pinn::geometry
