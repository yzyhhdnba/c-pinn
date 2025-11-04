#pragma once

#include <filesystem>
#include <string>

#include "pinn/nn/fnn.hpp"
#include "pinn/utils/config.hpp"

namespace pinn::utils {

class CheckpointManager {
  public:
    explicit CheckpointManager(CheckpointConfig config);

    void save(const nn::Fnn& network, int epoch, double loss);
    void load_latest(nn::Fnn& network);

  private:
    std::filesystem::path checkpoint_dir_;
    int save_every_;
};

}  // namespace pinn::utils
