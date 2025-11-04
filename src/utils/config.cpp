#include "pinn/utils/config.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "pinn/utils/logger.hpp"

namespace pinn::utils {

namespace {
std::vector<int> read_optional_int_vector(const nlohmann::json& node, const char* key) {
    if (!node.contains(key)) {
        return {};
    }
    return node.at(key).get<std::vector<int>>();
}

std::string read_optional_string(const nlohmann::json& node, const char* key, const std::string& fallback) {
    if (node.contains(key)) {
        return node.at(key).get<std::string>();
    }
    return fallback;
}

int read_optional_int(const nlohmann::json& node, const char* key, int fallback) {
    if (node.contains(key)) {
        return node.at(key).get<int>();
    }
    return fallback;
}

double read_optional_double(const nlohmann::json& node, const char* key, double fallback) {
    if (node.contains(key)) {
        return node.at(key).get<double>();
    }
    return fallback;
}

}  // namespace

ConfigBundle load_config(const std::filesystem::path& path) {
    auto absolute_path = std::filesystem::absolute(path);
    std::ifstream stream{absolute_path};
    if (!stream) {
        throw std::runtime_error{"Failed to open config file: " + absolute_path.string()};
    }

    nlohmann::json json;
    stream >> json;

    ConfigBundle bundle;
    bundle.raw = json;

    if (json.contains("model")) {
        const auto& model = json.at("model");
        bundle.network.input_dim = model.value("input_dim", bundle.network.input_dim);
        bundle.network.output_dim = model.value("output_dim", bundle.network.output_dim);
        bundle.network.hidden_layers = read_optional_int_vector(model, "layers");
        bundle.network.activation = read_optional_string(model, "activation", bundle.network.activation);
        bundle.network.weight_init = read_optional_string(model, "weight_init", bundle.network.weight_init);
        bundle.network.bias_init = model.value("bias_init", bundle.network.bias_init);
        bundle.network.seed = model.value("seed", bundle.network.seed);
    }

    if (json.contains("training")) {
        const auto& training = json.at("training");
        bundle.training.optimizer = read_optional_string(training, "optimizer", bundle.training.optimizer);
        bundle.training.learning_rate = training.value("lr", bundle.training.learning_rate);
        bundle.training.batch_size = training.value("batch_size", bundle.training.batch_size);
        bundle.training.epochs = training.value("epochs", bundle.training.epochs);
        if (training.contains("lr_schedule")) {
            const auto& schedule = training.at("lr_schedule");
            bundle.training.milestones = read_optional_int_vector(schedule, "milestones");
            bundle.training.gamma = schedule.value("gamma", bundle.training.gamma);
        }
        bundle.training.use_lbfgs_after = training.value("use_lbfgs_after", bundle.training.use_lbfgs_after);
    }

    if (json.contains("data")) {
        const auto& data = json.at("data");
        bundle.data.n_interior = data.value("n_interior", bundle.data.n_interior);
        bundle.data.n_boundary = data.value("n_boundary", bundle.data.n_boundary);
        bundle.data.sampling = read_optional_string(data, "sampling", bundle.data.sampling);
    }

    if (json.contains("ad")) {
        const auto& ad = json.at("ad");
        bundle.ad.backend = read_optional_string(ad, "backend", bundle.ad.backend);
        bundle.ad.mode = read_optional_string(ad, "mode", bundle.ad.mode);
        bundle.ad.max_tape_size_mb = ad.value("max_tape_size_mb", bundle.ad.max_tape_size_mb);
    }

    if (json.contains("checkpoint")) {
        const auto& checkpoint = json.at("checkpoint");
        if (checkpoint.contains("dir")) {
            bundle.checkpoint.directory = std::filesystem::path{checkpoint.at("dir").get<std::string>()};
        }
        bundle.checkpoint.save_every = checkpoint.value("save_every", bundle.checkpoint.save_every);
    }

    Logger::instance().info("Loaded configuration from " + absolute_path.string());

    return bundle;
}

}  // namespace pinn::utils
