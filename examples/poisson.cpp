#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "pinn/geometry/interval.hpp"
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

int main(int argc, char** argv) {
    using namespace pinn;
    namespace fs = std::filesystem;

    std::vector<fs::path> candidate_paths;
    if (const char* env_path = std::getenv("PINN_CONFIG"); env_path != nullptr && env_path[0] != '\0') {
        candidate_paths.emplace_back(env_path);
    }
    if (argc > 1) {
        candidate_paths.emplace_back(argv[1]);
    }
    candidate_paths.emplace_back("config/pinn_config.json");
    candidate_paths.emplace_back("../config/pinn_config.json");
    const fs::path binary_dir = fs::absolute(fs::path(argv[0])).parent_path();
    candidate_paths.emplace_back(binary_dir / "../config/pinn_config.json");
    candidate_paths.emplace_back(binary_dir / "../../config/pinn_config.json");

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

    geometry::Interval domain{0.0, 1.0};

    auto residual_fn = [](const pde::DifferentialData& data) {
        using namespace torch::indexing;
        auto x = data.point.index({Slice(), 0});
        auto u_xx = data.hessian.index({Slice(), 0, 0});
        auto forcing = (M_PI * M_PI) * torch::sin(M_PI * x);
        return u_xx + forcing;
    };

    auto pde_ptr = std::make_shared<pde::Pde>(domain, residual_fn);

    auto boundary_fn = [](const Tensor& points) {
        return torch::zeros({points.size(0), 1}, points.options());
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

    TrainingBatch batch;
    auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat).device(device);
    batch.interior_points = geometry::sample_interior(domain, config.data.n_interior, sampling, generator).to(tensor_opts);
    batch.boundary_points = geometry::sample_boundary(domain, config.data.n_boundary, generator).to(tensor_opts);
    batch.boundary_values = torch::zeros({batch.boundary_points.size(0), 1}, tensor_opts);

    model::TrainingOptions options;
    options.epochs = config.training.epochs;
    options.batch_size = config.training.batch_size;
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

    model::Trainer trainer(pinn_model, options);
    trainer.fit(batch, callbacks);

    return 0;
}
