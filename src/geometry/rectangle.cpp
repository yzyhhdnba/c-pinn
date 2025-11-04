#include "pinn/geometry/rectangle.hpp"

#include <cmath>
#include <stdexcept>

namespace pinn::geometry {

namespace {
Tensor make_tensor(const std::vector<Scalar>& values) {
    auto opts = torch::dtype(torch::kDouble);
    return torch::tensor(values, opts);
}

Tensor ensure_vector_shape(const Tensor& x, int dim) {
    if (x.numel() != dim) {
        TORCH_CHECK(false, "Expected tensor with %d elements, got %ld", dim, x.numel());
    }
    return x.reshape({dim}).to(torch::kDouble);
}
}  // namespace

Rectangle::Rectangle(const std::vector<Scalar>& lower, const std::vector<Scalar>& upper)
    : Geometry(static_cast<int>(lower.size())), lower_{make_tensor(lower)}, upper_{make_tensor(upper)} {
    if (lower.size() != upper.size()) {
        throw std::invalid_argument{"Lower/upper bounds must have same dimensionality"};
    }
    if (lower.empty()) {
        throw std::invalid_argument{"Rectangle requires at least one dimension"};
    }
    if ((upper_ <= lower_).any().item<bool>()) {
        throw std::invalid_argument{"Upper bounds must exceed lower bounds in every dimension"};
    }
}

bool Rectangle::inside(const Tensor& x) const {
    auto flat = ensure_vector_shape(x, dimension());
    auto gt_lower = (flat > lower_).all().item<bool>();
    auto lt_upper = (flat < upper_).all().item<bool>();
    return gt_lower && lt_upper;
}

bool Rectangle::on_boundary(const Tensor& x) const {
    const double tol = 1e-12;
    auto flat = ensure_vector_shape(x, dimension());
    for (int i = 0; i < dimension(); ++i) {
        const double val = flat[i].item<Scalar>();
        const double lo = lower_[i].item<Scalar>();
        const double hi = upper_[i].item<Scalar>();
        if (std::abs(val - lo) < tol || std::abs(val - hi) < tol) {
            return true;
        }
    }
    return false;
}

Tensor Rectangle::boundary_normal(const Tensor& x) const {
    const double tol = 1e-12;
    auto flat = ensure_vector_shape(x, dimension());
    auto normal = torch::zeros({dimension()}, torch::dtype(torch::kDouble));
    for (int i = 0; i < dimension(); ++i) {
        const double val = flat[i].item<Scalar>();
        const double lo = lower_[i].item<Scalar>();
        const double hi = upper_[i].item<Scalar>();
        if (std::abs(val - lo) < tol) {
            normal[i] = -1.0;
        } else if (std::abs(val - hi) < tol) {
            normal[i] = 1.0;
        }
    }
    const auto norm = normal.norm().item<Scalar>();
    if (norm > 0.0) {
        normal = normal / norm;
    }
    return normal;
}

Tensor Rectangle::uniform_points(int n_per_dim) const {
    TORCH_CHECK(n_per_dim > 1, "Uniform grid requires at least two samples per dimension");
    std::vector<Tensor> grids;
    grids.reserve(dimension());
    for (int i = 0; i < dimension(); ++i) {
        auto lin = torch::linspace(lower_[i].item<Scalar>(), upper_[i].item<Scalar>(), n_per_dim,
                                   torch::dtype(torch::kDouble));
        grids.push_back(lin);
    }
    auto mesh = torch::meshgrid(grids, "ij");
    auto stacked = torch::stack(mesh, 0);
    return stacked.reshape({dimension(), -1}).transpose(0, 1).contiguous();
}

Tensor Rectangle::random_points(int n, torch::Generator& gen) const {
    (void)gen;
    auto rand = torch::rand({n, dimension()}, torch::dtype(torch::kDouble));
    auto span = upper_ - lower_;
    return rand * span + lower_;
}

std::pair<Tensor, Tensor> Rectangle::bounds() const {
    return {lower_, upper_};
}

}  // namespace pinn::geometry
