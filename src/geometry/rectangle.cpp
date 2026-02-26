#include "pinn/geometry/rectangle.hpp"

#include <cmath>
#include <stdexcept>

namespace pinn::geometry {

namespace {
Tensor make_tensor(const std::vector<Scalar>& values) {
    Tensor t({static_cast<int64_t>(values.size())});
    auto* p = t.data_ptr<double>();
    for (size_t i = 0; i < values.size(); ++i) {
        p[i] = static_cast<double>(values[i]);
    }
    return t;
}

Tensor ensure_vector_shape(const Tensor& x, int dim) {
    if (x.numel() != dim) {
        throw std::invalid_argument{"Expected tensor with " + std::to_string(dim) +
                                    " elements, got " + std::to_string(x.numel())};
    }
    return x.reshape({dim});
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
    for (int i = 0; i < dimension(); ++i) {
        const double lo = lower_.at(i);
        const double hi = upper_.at(i);
        if (!(hi > lo)) {
            throw std::invalid_argument{"Upper bounds must exceed lower bounds in every dimension"};
        }
    }
}

bool Rectangle::inside(const Tensor& x) const {
    auto flat = ensure_vector_shape(x, dimension());
    for (int i = 0; i < dimension(); ++i) {
        const double v = flat.at(i);
        const double lo = lower_.at(i);
        const double hi = upper_.at(i);
        if (!(v > lo && v < hi)) {
            return false;
        }
    }
    return true;
}

bool Rectangle::on_boundary(const Tensor& x) const {
    const double tol = 1e-12;
    auto flat = ensure_vector_shape(x, dimension());
    for (int i = 0; i < dimension(); ++i) {
        const double val = flat.at(i);
        const double lo = lower_.at(i);
        const double hi = upper_.at(i);
        if (std::abs(val - lo) < tol || std::abs(val - hi) < tol) {
            return true;
        }
    }
    return false;
}

Tensor Rectangle::boundary_normal(const Tensor& x) const {
    const double tol = 1e-12;
    auto flat = ensure_vector_shape(x, dimension());
    auto normal = pinn::core::Tensor::zeros({dimension()});
    for (int i = 0; i < dimension(); ++i) {
        const double val = flat.at(i);
        const double lo = lower_.at(i);
        const double hi = upper_.at(i);
        if (std::abs(val - lo) < tol) {
            normal.set(i, -1.0);
        } else if (std::abs(val - hi) < tol) {
            normal.set(i, 1.0);
        }
    }
    const double nrm = normal.norm(2).item<double>();
    if (nrm > 0.0) {
        auto* p = normal.data_ptr<double>();
        for (int64_t i = 0; i < normal.numel(); ++i) {
            p[i] /= nrm;
        }
    }
    return normal;
}

Tensor Rectangle::uniform_points(int n_per_dim) const {
    if (n_per_dim <= 1) {
        throw std::invalid_argument{"Uniform grid requires at least two samples per dimension"};
    }
    const int dim = dimension();
    int64_t total = 1;
    for (int i = 0; i < dim; ++i) {
        total *= n_per_dim;
    }

    Tensor out({total, dim});
    for (int64_t idx = 0; idx < total; ++idx) {
        int64_t t = idx;
        for (int d = dim - 1; d >= 0; --d) {
            const int64_t pos = t % n_per_dim;
            t /= n_per_dim;
            const double lo = lower_.at(d);
            const double hi = upper_.at(d);
            const double alpha = static_cast<double>(pos) / static_cast<double>(n_per_dim - 1);
            out.set(idx, d, lo + (hi - lo) * alpha);
        }
    }
    return out;
}

Tensor Rectangle::random_points(int n, pinn::core::Rng& rng) const {
    const int dim = dimension();
    Tensor out({n, dim});
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < dim; ++d) {
            const double u = rng.uniform01();
            const double lo = lower_.at(d);
            const double hi = upper_.at(d);
            out.set(i, d, lo + (hi - lo) * u);
        }
    }
    return out;
}

std::pair<Tensor, Tensor> Rectangle::bounds() const {
    return {lower_, upper_};
}

}  // namespace pinn::geometry
