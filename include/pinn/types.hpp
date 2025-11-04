#pragma once

#include <torch/torch.h>

#include <vector>

namespace pinn {

using Tensor = torch::Tensor;
using TensorList = std::vector<Tensor>;
using Scalar = double;

struct TrainingBatch {
    Tensor interior_points;
    Tensor boundary_points;
    Tensor boundary_values;
};

}  // namespace pinn
