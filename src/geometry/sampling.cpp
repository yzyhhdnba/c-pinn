#include "pinn/geometry/sampling.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
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
    auto left = torch::full({half, 1}, interval.left(), torch::dtype(torch::kDouble));
    auto right = torch::full({n_points - half, 1}, interval.right(), torch::dtype(torch::kDouble));
    return torch::cat({left, right}, 0);
}

Tensor sample_rectangle_boundary(const Rectangle& rectangle, int n_points, torch::Generator& gen) {
    (void)gen;
    const int dim = rectangle.dimension();
    auto lower_upper = rectangle.bounds();
    auto lower = lower_upper.first;
    auto upper = lower_upper.second;
    auto opts = torch::dtype(torch::kDouble);
    auto points = torch::zeros({n_points, dim}, opts);
    auto rand = torch::rand({n_points, dim}, opts);
    points = rand * (upper - lower) + lower;
    auto faces = torch::randint(0, dim * 2, {n_points}, torch::dtype(torch::kLong));
    for (int i = 0; i < n_points; ++i) {
        const int face = faces[i].item<int>();
        const int axis = face / 2;
        const bool is_upper = (face % 2) == 1;
        points[i][axis] = is_upper ? upper[axis] : lower[axis];
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

Tensor sample_interior(const Geometry& geometry, int n_points, SamplingStrategy strategy, torch::Generator& gen) {
    switch (strategy) {
        case SamplingStrategy::kUniformGrid:
            return geometry.uniform_points(static_cast<int>(std::round(std::pow(n_points, 1.0 / geometry.dimension()))));
        case SamplingStrategy::kLatinHypercube:
        default:
            return geometry.random_points(n_points, gen);
    }
}

Tensor sample_boundary(const Geometry& geometry, int n_points, torch::Generator& gen) {
    if (const auto* interval = dynamic_cast<const Interval*>(&geometry)) {
        return sample_interval_boundary(*interval, n_points);
    }
    if (const auto* rectangle = dynamic_cast<const Rectangle*>(&geometry)) {
        return sample_rectangle_boundary(*rectangle, n_points, gen);
    }
    throw std::invalid_argument{"Boundary sampling not implemented for this geometry"};
}

}  // namespace pinn::geometry
