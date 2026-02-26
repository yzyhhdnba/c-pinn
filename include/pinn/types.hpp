#pragma once

#include "pinn/core/tensor.hpp"

#include <vector>

namespace pinn {

using Tensor = core::Tensor;
using TensorList = std::vector<Tensor>;
using Scalar = double;

struct TrainingBatch {
    Tensor interior_points;
    Tensor boundary_points;
    Tensor boundary_values;
};

}  // namespace pinn
