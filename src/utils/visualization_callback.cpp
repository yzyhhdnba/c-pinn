#include "pinn/utils/visualization_callback.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

#include <torch/torch.h>

#include "pinn/utils/logger.hpp"

namespace pinn::utils {

namespace {
std::string sanitize_name(const std::string& name) {
    if (!name.empty()) {
        return name;
    }
    return "visual";
}
}  // namespace

VisualizationCallback::VisualizationCallback(model::Model& model,
                                             std::vector<VisualizationSpec> specs,
                                             VisualizationOptions options)
    : model_{&model},
      specs_{std::move(specs)},
      options_{std::move(options)} {
    if (options_.interval <= 0) {
        options_.interval = 1;
    }
    if (!options_.output_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(options_.output_dir, ec);
        if (ec) {
            Logger::instance().warn("Failed to create visualization directory: " + options_.output_dir.string() + ", " + ec.message());
        }
    }
}

void VisualizationCallback::on_epoch_end(const CallbackContext& ctx) {
    if (specs_.empty()) {
        return;
    }
    if (options_.interval <= 0) {
        return;
    }
    if (ctx.epoch % options_.interval != 0) {
        return;
    }

    auto& network = model_->network();
    torch::NoGradGuard no_grad;

    torch::Device model_device = torch::kCPU;
    auto params = network->parameters();
    if (!params.empty()) {
        model_device = params.front().device();
    }

    for (const auto& spec : specs_) {
        if (!spec.points.defined() || spec.points.numel() == 0) {
            continue;
        }

        torch::Tensor points_device = spec.points.to(model_device, /*non_blocking=*/true);
        torch::Tensor predictions = network->forward(points_device);
        torch::Tensor predictions_cpu = predictions.to(torch::kCPU, /*non_blocking=*/true);
        torch::Tensor points_cpu = spec.points.to(torch::kCPU, /*non_blocking=*/true);

        torch::Tensor reference_cpu;
        torch::Tensor abs_err_cpu;
        if (spec.reference) {
            reference_cpu = spec.reference(points_cpu);
            if (reference_cpu.defined()) {
                if (reference_cpu.device().is_cuda()) {
                    reference_cpu = reference_cpu.to(torch::kCPU, /*non_blocking=*/true);
                }
                if (reference_cpu.dim() == 1) {
                    reference_cpu = reference_cpu.unsqueeze(1);
                }
                if (predictions_cpu.dim() == 1) {
                    predictions_cpu = predictions_cpu.unsqueeze(1);
                }
                reference_cpu = reference_cpu.reshape_as(predictions_cpu);
                abs_err_cpu = (predictions_cpu - reference_cpu).abs();
            }
        }

        std::ostringstream filename;
        filename << sanitize_name(spec.name) << "_epoch_" << std::setw(5) << std::setfill('0') << ctx.epoch << ".csv";
        auto path = options_.output_dir / filename.str();
        write_csv(path,
                  points_cpu,
                  predictions_cpu,
                  reference_cpu.defined() ? &reference_cpu : nullptr,
                  abs_err_cpu.defined() ? &abs_err_cpu : nullptr);
        Logger::instance().info("Visualization exported to " + path.string());
    }
}

void VisualizationCallback::write_csv(const std::filesystem::path& path,
                                      const Tensor& points_cpu,
                                      const Tensor& predictions_cpu,
                                      const Tensor* reference_cpu,
                                      const Tensor* abs_error_cpu) const {
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
        Logger::instance().warn("Failed to open visualization file: " + path.string());
        return;
    }

    torch::Tensor pts = points_cpu;
    if (pts.dim() == 1) {
        pts = pts.unsqueeze(1);
    }
    if (pts.dtype() != torch::kDouble) {
        pts = pts.to(torch::kDouble);
    }
    pts = pts.contiguous();

    torch::Tensor preds = predictions_cpu;
    if (preds.dim() == 1) {
        preds = preds.unsqueeze(1);
    }
    if (preds.dtype() != torch::kDouble) {
        preds = preds.to(torch::kDouble);
    }
    preds = preds.contiguous();

    torch::Tensor ref;
    if (reference_cpu != nullptr && reference_cpu->defined()) {
        ref = *reference_cpu;
        if (ref.dim() == 1) {
            ref = ref.unsqueeze(1);
        }
        if (ref.dtype() != torch::kDouble) {
            ref = ref.to(torch::kDouble);
        }
        ref = ref.contiguous();
    }

    torch::Tensor abs_err;
    if (abs_error_cpu != nullptr && abs_error_cpu->defined()) {
        abs_err = *abs_error_cpu;
        if (abs_err.dim() == 1) {
            abs_err = abs_err.unsqueeze(1);
        }
        if (abs_err.dtype() != torch::kDouble) {
            abs_err = abs_err.to(torch::kDouble);
        }
        abs_err = abs_err.contiguous();
    }

    const int64_t rows = pts.size(0);
    const int64_t dims = pts.size(1);
    const int64_t outputs = preds.size(1);
    const bool has_reference = ref.defined();
    const bool has_error = abs_err.defined();

    stream << std::setprecision(10);
    if (options_.write_header) {
        bool first = true;
        for (int64_t i = 0; i < dims; ++i) {
            if (!first) {
                stream << ',';
            }
            stream << 'x' << i;
            first = false;
        }
        for (int64_t i = 0; i < outputs; ++i) {
            stream << ",pred" << i;
        }
        if (has_reference) {
            for (int64_t i = 0; i < outputs; ++i) {
                stream << ",target" << i;
            }
        }
        if (has_error) {
            for (int64_t i = 0; i < outputs; ++i) {
                stream << ",abs_error" << i;
            }
        }
        stream << '\n';
    }

    for (int64_t r = 0; r < rows; ++r) {
        bool first = true;
        for (int64_t c = 0; c < dims; ++c) {
            if (!first) {
                stream << ',';
            }
            stream << pts.index({r, c}).item<double>();
            first = false;
        }
        for (int64_t c = 0; c < outputs; ++c) {
            stream << ',' << preds.index({r, c}).item<double>();
        }
        if (has_reference) {
            for (int64_t c = 0; c < outputs; ++c) {
                stream << ',' << ref.index({r, c}).item<double>();
            }
        }
        if (has_error) {
            for (int64_t c = 0; c < outputs; ++c) {
                stream << ',' << abs_err.index({r, c}).item<double>();
            }
        }
        stream << '\n';
    }
}

}  // namespace pinn::utils
