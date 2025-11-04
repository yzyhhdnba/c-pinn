#include "pinn/utils/checkpoint.hpp"

#include <algorithm>
#include <filesystem>
#include <regex>
#include <stdexcept>

#include <torch/torch.h>

#include "pinn/utils/logger.hpp"

namespace pinn::utils {

CheckpointManager::CheckpointManager(CheckpointConfig config)
    : checkpoint_dir_{std::move(config.directory)}, save_every_{config.save_every} {
    if (save_every_ <= 0) {
        throw std::invalid_argument{"Checkpoint save interval must be positive"};
    }
    std::filesystem::create_directories(checkpoint_dir_);
}

void CheckpointManager::save(const nn::Fnn& network, int epoch, double loss) {
    if (epoch % save_every_ != 0) {
        return;
    }
    auto filename = checkpoint_dir_ / ("epoch_" + std::to_string(epoch) + "_" + std::to_string(loss) + ".pt");
    torch::save(network, filename.string());
    Logger::instance().info("Saved checkpoint to " + filename.string());
}

void CheckpointManager::load_latest(nn::Fnn& network) {
    if (!std::filesystem::exists(checkpoint_dir_)) {
        return;
    }

    std::regex pattern{"epoch_([0-9]+)_.*\\.pt"};
    std::filesystem::path latest;
    int best_epoch = -1;

    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir_)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::smatch match;
        const auto filename = entry.path().filename().string();
        if (std::regex_match(filename, match, pattern)) {
            int epoch = std::stoi(match[1]);
            if (epoch > best_epoch) {
                best_epoch = epoch;
                latest = entry.path();
            }
        }
    }

    if (best_epoch >= 0) {
        torch::load(network, latest.string());
        Logger::instance().info("Loaded checkpoint from " + latest.string());
    }
}

}  // namespace pinn::utils
