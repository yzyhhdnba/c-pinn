#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace pinn::utils {

struct NetworkConfig {
    int input_dim{1};
    int output_dim{1};
    std::vector<int> hidden_layers{};
    std::string activation{"tanh"};
    std::string weight_init{"xavier_uniform"};
    double bias_init{0.0};
    int seed{42};
};

struct TrainingConfig {
    std::string optimizer{"adam"};
    double learning_rate{1e-3};
    int batch_size{1024};
    int epochs{20000};
    std::vector<int> milestones{};
    double gamma{0.1};
    int use_lbfgs_after{-1};
};

struct DataConfig {
    int n_interior{10000};
    int n_boundary{2000};
    std::string sampling{"latin_hypercube"};
};

struct AdConfig {
    std::string backend{"libtorch"};
    std::string mode{"reverse"};
    int max_tape_size_mb{1024};
};

struct CheckpointConfig {
    std::filesystem::path directory{"./ckpt"};
    int save_every{1000};
};

struct ConfigBundle {
    NetworkConfig network;
    TrainingConfig training;
    DataConfig data;
    AdConfig ad;
    CheckpointConfig checkpoint;
    nlohmann::json raw;
};

ConfigBundle load_config(const std::filesystem::path& path);

}  // namespace pinn::utils
