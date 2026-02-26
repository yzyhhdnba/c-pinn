#include "pinn/geometry/sampling.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include "pinn/geometry/interval.hpp"
#include "pinn/geometry/rectangle.hpp"

namespace pinn::geometry {

namespace {
std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

Tensor sample_interval_boundary(const Interval& interval, int n_points) {
    const int half = n_points / 2;
    auto left = pinn::core::Tensor::full({half, 1}, static_cast<double>(interval.left()));
    auto right = pinn::core::Tensor::full({n_points - half, 1}, static_cast<double>(interval.right()));
    return pinn::core::Tensor::cat({left, right}, 0);
}

Tensor sample_rectangle_boundary(const Rectangle& rectangle, int n_points, pinn::core::Rng& rng) {
    const int dim = rectangle.dimension();
    auto lower_upper = rectangle.bounds();
    auto lower = lower_upper.first;
    auto upper = lower_upper.second;

    Tensor points({n_points, dim});
    for (int64_t i = 0; i < n_points; ++i) {
        for (int d = 0; d < dim; ++d) {
            const double u = rng.uniform01();
            const double lo = lower.at(d);
            const double hi = upper.at(d);
            points.set(i, d, lo + (hi - lo) * u);
        }

        const int face = static_cast<int>(rng.randint(0, dim * 2));
        const int axis = face / 2;
        const bool is_upper = (face % 2) == 1;
        points.set(i, axis, is_upper ? upper.at(axis) : lower.at(axis));
    }
    return points;
}

}  // namespace

SamplingStrategy sampling_strategy_from_string(const std::string& name) {
    const auto lower = to_lower(name);
    if (lower == "uniform" || lower == "uniform_grid") {
        return SamplingStrategy::kUniformGrid;
    }
    if (lower == "latin" || lower == "latin_hypercube") {
        return SamplingStrategy::kLatinHypercube;
    }
    throw std::invalid_argument{"Unsupported sampling strategy: " + name};
}

Tensor sample_interior(const Geometry& geometry, int n_points, SamplingStrategy strategy, pinn::core::Rng& rng) {
    switch (strategy) {
        case SamplingStrategy::kUniformGrid:
            return geometry.uniform_points(static_cast<int>(std::round(std::pow(n_points, 1.0 / geometry.dimension()))));
        case SamplingStrategy::kLatinHypercube:
        default:
            return geometry.random_points(n_points, rng);
    }
}

Tensor sample_boundary(const Geometry& geometry, int n_points, pinn::core::Rng& rng) {
    if (const auto* interval = dynamic_cast<const Interval*>(&geometry)) {
        return sample_interval_boundary(*interval, n_points);
    }
    if (const auto* rectangle = dynamic_cast<const Rectangle*>(&geometry)) {
        return sample_rectangle_boundary(*rectangle, n_points, rng);
    }
    throw std::invalid_argument{"Boundary sampling not implemented for this geometry"};
}

}  // namespace pinn::geometry
