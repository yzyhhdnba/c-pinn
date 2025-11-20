#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "pinn/geometry/rectangle.hpp"
#include "pinn/geometry/sampling.hpp"
#include "pinn/model/model.hpp"
#include "pinn/model/trainer.hpp"
#include "pinn/nn/activation.hpp"
#include "pinn/nn/fnn.hpp"
#include "pinn/nn/initialization.hpp"
#include "pinn/pde/boundary_condition.hpp"
#include "pinn/pde/pde.hpp"
#include "pinn/types.hpp"
#include "pinn/utils/callback.hpp"
#include "pinn/utils/config.hpp"
#include "pinn/utils/visualization_callback.hpp"

int main(int argc, char** argv) {
    using namespace pinn;
    namespace fs = std::filesystem;
    using namespace torch::indexing;

    std::vector<fs::path> candidate_paths;
    if (const char* env_path = std::getenv("PINN_CONFIG"); env_path != nullptr && env_path[0] != '\0') {
        candidate_paths.emplace_back(env_path);
    }
    if (argc > 1) {
        candidate_paths.emplace_back(argv[1]);
    }
    candidate_paths.emplace_back("config/advection_config.json");
    candidate_paths.emplace_back("../config/advection_config.json");
    const fs::path binary_dir = fs::absolute(fs::path(argv[0])).parent_path();
    candidate_paths.emplace_back(binary_dir / "../config/advection_config.json");
    candidate_paths.emplace_back(binary_dir / "../../config/advection_config.json");

    fs::path config_path;
    for (const auto& candidate : candidate_paths) {
        if (candidate.empty()) {
            continue;
        }
        auto absolute = fs::absolute(candidate);
        if (fs::exists(absolute)) {
            config_path = absolute;
            break;
        }
    }

    if (config_path.empty()) {
        std::cerr << "无法找到配置文件，请通过参数或 PINN_CONFIG 指定。" << std::endl;
        return 1;
    }

    auto config = utils::load_config(config_path);
    std::cout << "Loaded config: " << config_path << std::endl;

    double velocity = 1.0;
    if (config.raw.contains("pde")) {
        const auto& pde_config = config.raw.at("pde");
        if (pde_config.contains("velocity")) {
            velocity = pde_config.at("velocity").get<double>();
        }
    }

    const int64_t seed = config.network.seed;

    const bool cuda_available = torch::cuda::is_available();
    std::cout << "torch::cuda::is_available(): " << (cuda_available ? "true" : "false") << std::endl;
    torch::Device device = cuda_available ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    torch::manual_seed(seed);
    if (device.is_cuda()) {
        torch::cuda::manual_seed_all(seed);
        std::cout << "Running on CUDA device 0" << std::endl;
    } else {
        std::cout << "CUDA not available, falling back to CPU" << std::endl;
    }

    torch::Generator generator = torch::make_generator<torch::CPUGeneratorImpl>(static_cast<uint64_t>(seed));

    geometry::Rectangle domain({0.0, 0.0}, {1.0, 1.0});

    auto residual_fn = [velocity](const pde::DifferentialData& data) {
        auto u_t = data.gradients.index({Slice(), 1});
        auto u_x = data.gradients.index({Slice(), 0});
        return u_t + velocity * u_x;
    };

    auto pde_ptr = std::make_shared<pde::Pde>(domain, residual_fn);

    auto boundary_fn = [velocity](const Tensor& points) {
        auto x = points.index({Slice(), 0});
        auto t = points.index({Slice(), 1});
        auto values = torch::sin(M_PI * (x - velocity * t));
        return values.unsqueeze(1);
    };

    auto bc = std::make_shared<pde::DirichletBC>(domain, boundary_fn);
    std::vector<std::shared_ptr<pde::BoundaryCondition>> bcs{bc};

    std::vector<int> layers;
    layers.reserve(config.network.hidden_layers.size() + 2);
    layers.push_back(config.network.input_dim);
    layers.insert(layers.end(), config.network.hidden_layers.begin(), config.network.hidden_layers.end());
    layers.push_back(config.network.output_dim);

    auto network = std::make_shared<nn::Fnn>(layers,
                                             nn::activation_from_string(config.network.activation),
                                             nn::init_from_string(config.network.weight_init),
                                             static_cast<Scalar>(config.network.bias_init),
                                             seed);
    (*network)->to(torch::kFloat);
    (*network)->to(device);

    model::LossWeights weights;
    weights.data = 0.0;
    auto pinn_model = model::Model(network, pde_ptr, bcs, weights);

    geometry::SamplingStrategy sampling = geometry::sampling_strategy_from_string(config.data.sampling);

    model::TrainingOptions options;
    options.epochs = config.training.epochs;
    options.batch_size = config.data.n_interior;
    options.mini_batch_size = config.training.batch_size;
    options.schedule.optimizer_name = config.training.optimizer;
    options.schedule.learning_rate = static_cast<Scalar>(config.training.learning_rate);
    options.schedule.milestones = config.training.milestones;
    options.schedule.gamma = static_cast<Scalar>(config.training.gamma);
    options.schedule.switch_to_lbfgs_epoch = config.training.use_lbfgs_after;

    utils::CallbackRegistry callbacks;
    struct PrintCallback : utils::Callback {
        void on_epoch_end(const utils::CallbackContext& ctx) override {
            std::cout << "Epoch " << ctx.epoch << ": loss=" << ctx.loss << std::endl;
        }
    };
    callbacks.add(std::make_shared<PrintCallback>());

    int grid = 64;
    auto x_eval = torch::linspace(0.0, 1.0, grid, torch::TensorOptions().dtype(torch::kFloat));
    auto t_eval = torch::linspace(0.0, 1.0, grid, torch::TensorOptions().dtype(torch::kFloat));
    auto mesh = torch::meshgrid({x_eval, t_eval});
    auto eval_points = torch::stack({mesh[0].reshape({-1}), mesh[1].reshape({-1})}, 1);

    utils::VisualizationOptions viz_options;
    viz_options.output_dir = fs::path{"sandbox"} / "advection";
    int viz_interval = config.training.epochs / 10;
    if (viz_interval <= 0) {
        viz_interval = 1;
    }
    viz_options.interval = viz_interval;

    utils::VisualizationSpec viz_spec;
    viz_spec.name = "advection";
    viz_spec.points = eval_points;
    viz_spec.reference = [velocity](const Tensor& points) {
        auto x = points.index({Slice(), 0});
        auto t = points.index({Slice(), 1});
        auto values = torch::sin(M_PI * (x - velocity * t));
        return values.unsqueeze(1);
    };

    callbacks.add(std::make_shared<utils::VisualizationCallback>(pinn_model, std::vector<utils::VisualizationSpec>{viz_spec}, viz_options));

    model::Trainer trainer(pinn_model, options);
    
    // 使用重采样接口
    auto sampler = [&](const torch::Device& dev, torch::Generator& gen) -> TrainingBatch {
        TrainingBatch batch;
        auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat).device(dev);
        batch.interior_points = geometry::sample_interior(domain, config.data.n_interior, sampling, gen).to(tensor_opts);
        batch.boundary_points = geometry::sample_boundary(domain, config.data.n_boundary, gen).to(tensor_opts);
        batch.boundary_values = boundary_fn(batch.boundary_points);
        return batch;
    };
    
    trainer.fit_with_resampling(sampler, generator, device, callbacks);

    return 0;
}
