#pragma once

#include <vector>

#include "pinn/geometry/geometry.hpp"

namespace pinn::geometry {

class Rectangle : public Geometry {
  public:
    Rectangle(const std::vector<Scalar>& lower, const std::vector<Scalar>& upper);

    bool inside(const Tensor& x) const override;
    bool on_boundary(const Tensor& x) const override;
    Tensor boundary_normal(const Tensor& x) const override;

    Tensor uniform_points(int n_per_dim) const override;
    Tensor random_points(int n, torch::Generator& gen) const override;

    std::pair<Tensor, Tensor> bounds() const override;

  private:
    Tensor lower_;
    Tensor upper_;
};

}  // namespace pinn::geometry
