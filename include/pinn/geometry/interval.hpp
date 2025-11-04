#pragma once

#include "pinn/geometry/geometry.hpp"

namespace pinn::geometry {

class Interval : public Geometry {
  public:
    Interval(Scalar left, Scalar right);

    bool inside(const Tensor& x) const override;
    bool on_boundary(const Tensor& x) const override;
    Tensor boundary_normal(const Tensor& x) const override;

    Tensor uniform_points(int n_per_dim) const override;
    Tensor random_points(int n, torch::Generator& gen) const override;

    std::pair<Tensor, Tensor> bounds() const override;

    Scalar left() const noexcept { return left_; }
    Scalar right() const noexcept { return right_; }

  private:
    Scalar left_;
    Scalar right_;
};

}  // namespace pinn::geometry
