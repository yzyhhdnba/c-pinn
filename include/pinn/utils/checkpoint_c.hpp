#pragma once

#include <filesystem>
#include <string>

#include "pinn/nn/fnn.hpp"

namespace pinn::utils {

// Pure C checkpoint manager (binary format)
class CheckpointManagerC {
  public:
    explicit CheckpointManagerC(const std::filesystem::path& directory, int save_every = 100);

    // Save network weights to binary file
    void save(const nn::Fnn& network, int epoch, double loss);

    // Load latest checkpoint into network
    void load_latest(nn::Fnn& network);

  private:
    std::filesystem::path checkpoint_dir_;
    int save_every_;

    std::string make_filename(int epoch, double loss) const;
    int parse_epoch_from_filename(const std::string& filename) const;
};

}  // namespace pinn::utils
