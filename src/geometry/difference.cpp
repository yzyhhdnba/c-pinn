#include "pinn/geometry/difference.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace pinn::geometry {
namespace {
Tensor gather_indices(const Tensor& candidates, const std::vector<int64_t>& indices) {
    if (indices.empty()) {
        return torch::empty({0, candidates.size(1)}, candidates.options());
    }
    auto idx_tensor = torch::tensor(indices, torch::TensorOptions().dtype(torch::kLong));
    return candidates.index_select(0, idx_tensor);
}
}  // namespace

Difference::Difference(std::shared_ptr<Geometry> primary,
                       std::shared_ptr<Geometry> subtract)
    : Geometry(primary ? primary->dimension() : 0),
      primary_{std::move(primary)},
      subtract_{std::move(subtract)} {
    if (primary_ == nullptr || subtract_ == nullptr) {
        throw std::invalid_argument{"Difference requires non-null geometries"};
    }
    if (primary_->dimension() != subtract_->dimension()) {
        throw std::invalid_argument{"Difference geometries must share dimensionality"};
    }
}

Tensor Difference::filter_valid(const Tensor& candidates) const {
    std::vector<int64_t> keep;
    keep.reserve(candidates.size(0));
    for (int64_t i = 0; i < candidates.size(0); ++i) {
        auto point = candidates.select(0, i);
        if (primary_->inside(point) && !subtract_->inside(point)) {
            keep.push_back(i);
        }
    }
    if (keep.empty()) {
        return torch::empty({0, candidates.size(1)}, candidates.options());
    }
    return gather_indices(candidates, keep);
}

bool Difference::inside(const Tensor& x) const {
    return primary_->inside(x) && !subtract_->inside(x);
}

bool Difference::on_boundary(const Tensor& x) const {
    const bool on_primary = primary_->on_boundary(x) && !subtract_->inside(x);
    const bool on_subtract = primary_->inside(x) && subtract_->on_boundary(x);
    return on_primary || on_subtract;
}

Tensor Difference::boundary_normal(const Tensor& x) const {
    if (primary_->on_boundary(x) && !subtract_->inside(x)) {
        return primary_->boundary_normal(x);
    }
    if (primary_->inside(x) && subtract_->on_boundary(x)) {
        return -subtract_->boundary_normal(x);
    }
    return torch::zeros({dimension()}, torch::dtype(torch::kDouble));
}

Tensor Difference::uniform_points(int n_per_dim) const {
    auto candidates = primary_->uniform_points(n_per_dim);
    auto filtered = filter_valid(candidates);
    if (filtered.size(0) == 0) {
        throw std::runtime_error("Difference::uniform_points produced no samples; consider increasing resolution");
    }
    return filtered;
}

Tensor Difference::random_points(int n, torch::Generator& gen) const {
    if (n <= 0) {
        return torch::empty({0, dimension()}, torch::dtype(torch::kDouble));
    }
    std::vector<Tensor> accepted;
    accepted.reserve(4);
    int collected = 0;
    int attempts = 0;
    const int max_attempts = 50;
    while (collected < n && attempts < max_attempts) {
        const int request = std::max(2 * (n - collected), 1);
        auto candidates = primary_->random_points(request, gen);
        auto filtered = filter_valid(candidates);
        if (filtered.size(0) > 0) {
            const int take = std::min<int>(filtered.size(0), n - collected);
            accepted.push_back(filtered.narrow(0, 0, take));
            collected += take;
        }
        ++attempts;
    }
    if (collected < n) {
        throw std::runtime_error("Difference::random_points failed to sample enough points; domain may be too small");
    }
    if (accepted.size() == 1) {
        return accepted.front();
    }
    return torch::cat(accepted, 0);
}

std::pair<Tensor, Tensor> Difference::bounds() const {
    return primary_->bounds();
}

}  // namespace pinn::geometry
