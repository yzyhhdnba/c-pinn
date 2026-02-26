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
        return pinn::core::Tensor::full({1}, -1.0);
    }
    if (std::abs(value - right_) < tol) {
        return pinn::core::Tensor::full({1}, 1.0);
    }
    return pinn::core::Tensor::zeros({1});
}

Tensor Interval::uniform_points(int n_per_dim) const {
    if (n_per_dim <= 1) {
        throw std::invalid_argument{"Uniform points require at least two samples"};
    }
    auto lin = pinn::core::Tensor::linspace(left_, right_, n_per_dim);
    return lin.unsqueeze(1);
}

Tensor Interval::random_points(int n, pinn::core::Rng& rng) const {
    auto r = pinn::core::Tensor::rand_uniform({n, 1}, rng);
    auto* p = r.data_ptr<double>();
    const double span = static_cast<double>(right_ - left_);
    for (int64_t i = 0; i < r.numel(); ++i) {
        p[i] = static_cast<double>(left_) + span * p[i];
    }
    return r;
}

std::pair<Tensor, Tensor> Interval::bounds() const {
    return {pinn::core::Tensor::full({1}, static_cast<double>(left_)),
            pinn::core::Tensor::full({1}, static_cast<double>(right_))};
}

}  // namespace pinn::geometry
