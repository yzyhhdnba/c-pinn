#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "pinn/model/model.hpp"
#include "pinn/types.hpp"
#include "pinn/utils/callback.hpp"

namespace pinn::utils {

struct VisualizationOptions {
    std::filesystem::path output_dir{"./sandbox"};
    int interval{100};
    bool write_header{true};
};

struct VisualizationSpec {
    std::string name;
    Tensor points;
    std::function<Tensor(const Tensor&)> reference{};
};

class VisualizationCallback : public Callback {
  public:
    VisualizationCallback(model::Model& model,
                          std::vector<VisualizationSpec> specs,
                          VisualizationOptions options = {});

    void on_epoch_end(const CallbackContext& ctx) override;

  private:
    model::Model* model_;
    std::vector<VisualizationSpec> specs_;
    VisualizationOptions options_;

    void write_csv(const std::filesystem::path& path,
                   const Tensor& points_cpu,
                   const Tensor& predictions_cpu,
                   const Tensor* reference_cpu,
                   const Tensor* abs_error_cpu) const;
};

}  // namespace pinn::utils
