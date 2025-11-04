#include "pinn/geometry/interval.hpp"

#include <cmath>
#include <stdexcept>

namespace pinn::geometry {

Interval::Interval(Scalar left, Scalar right)
    : Geometry(1), left_{left}, right_{right} {
    if (left_ >= right_) {
        throw std::invalid_argument{"Interval requires left < right"};
    }
}

bool Interval::inside(const Tensor& x) const {
    return x.item<Scalar>() > left_ && x.item<Scalar>() < right_;
}

bool Interval::on_boundary(const Tensor& x) const {
    const Scalar value = x.item<Scalar>();
    const Scalar tol = 1e-12;
    return std::abs(value - left_) < tol || std::abs(value - right_) < tol;
}

Tensor Interval::boundary_normal(const Tensor& x) const {
    const Scalar value = x.item<Scalar>();
    const Scalar tol = 1e-12;
    if (std::abs(value - left_) < tol) {
        return torch::tensor({-1.0}, torch::dtype(torch::kDouble));
    }
    if (std::abs(value - right_) < tol) {
        return torch::tensor({1.0}, torch::dtype(torch::kDouble));
    }
    return torch::zeros({1}, torch::dtype(torch::kDouble));
}

Tensor Interval::uniform_points(int n_per_dim) const {
    TORCH_CHECK(n_per_dim > 1, "Uniform points require at least two samples");
    auto lin = torch::linspace(left_, right_, n_per_dim, torch::dtype(torch::kDouble));
    return lin.unsqueeze(1);
}

Tensor Interval::random_points(int n, torch::Generator& gen) const {
    (void)gen;  // seed control can be added when wiring custom generators
    auto rand = torch::rand({n, 1}, torch::dtype(torch::kDouble));
    return left_ + (right_ - left_) * rand;
}

std::pair<Tensor, Tensor> Interval::bounds() const {
    auto opts = torch::dtype(torch::kDouble);
    return {torch::tensor({left_}, opts), torch::tensor({right_}, opts)};
}

}  // namespace pinn::geometry
