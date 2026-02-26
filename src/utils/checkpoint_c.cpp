#include "pinn/utils/checkpoint_c.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "pinn/nn/gemm.hpp"
#include "pinn/utils/logger.hpp"

namespace pinn::utils {

CheckpointManagerC::CheckpointManagerC(const std::filesystem::path& directory, int save_every)
    : checkpoint_dir_{directory}, save_every_{save_every} {
    if (save_every_ <= 0) {
        throw std::invalid_argument{"Checkpoint save interval must be positive"};
    }
    std::filesystem::create_directories(checkpoint_dir_);
}

std::string CheckpointManagerC::make_filename(int epoch, double loss) const {
    std::ostringstream oss;
    oss << "epoch_" << epoch << "_loss_" << loss << ".bin";
    return oss.str();
}

int CheckpointManagerC::parse_epoch_from_filename(const std::string& filename) const {
    std::regex pattern{"epoch_([0-9]+)_.*\\.bin"};
    std::smatch match;
    if (std::regex_match(filename, match, pattern) && match.size() > 1) {
        return std::stoi(match[1]);
    }
    return -1;
}

void CheckpointManagerC::save(const nn::Fnn& network, int epoch, double loss) {
    if (epoch % save_every_ != 0) {
        return;
    }

    auto filename = checkpoint_dir_ / make_filename(epoch, loss);
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open checkpoint file for writing: " + filename.string());
    }

    // Write header
    const int magic = 0x50494E4E;  // 'PINN'
    const int version = 1;
    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write network architecture
    const int num_layers = network.num_layers();
    ofs.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    const int input_dim = network.input_dim();
    ofs.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));

    for (int i = 0; i < num_layers; ++i) {
        const int out_features = network.layer(i).out_features();
        ofs.write(reinterpret_cast<const char*>(&out_features), sizeof(out_features));
    }

    // Write weights and biases - TODO: need flat accessors
    // For now, write layer by layer
    for (int i = 0; i < num_layers; ++i) {
        const auto& layer = network.layer(i);
        const nn::TYPE_VAL* w = layer.weight_data();
        const nn::TYPE_VAL* b = layer.bias_data();
        
        int w_count = layer.in_features() * layer.out_features();
        int b_count = layer.out_features();
        
        ofs.write(reinterpret_cast<const char*>(&w_count), sizeof(w_count));
        ofs.write(reinterpret_cast<const char*>(w), w_count * sizeof(nn::TYPE_VAL));
        
        ofs.write(reinterpret_cast<const char*>(&b_count), sizeof(b_count));
        ofs.write(reinterpret_cast<const char*>(b), b_count * sizeof(nn::TYPE_VAL));
    }

    ofs.close();
    Logger::instance().info("Saved checkpoint to " + filename.string());
}

void CheckpointManagerC::load_latest(nn::Fnn& network) {
    if (!std::filesystem::exists(checkpoint_dir_)) {
        return;
    }

    std::filesystem::path latest;
    int best_epoch = -1;

    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir_)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto filename = entry.path().filename().string();
        int epoch = parse_epoch_from_filename(filename);
        if (epoch > best_epoch) {
            best_epoch = epoch;
            latest = entry.path();
        }
    }

    if (best_epoch < 0) {
        return;
    }

    std::ifstream ifs(latest, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open checkpoint file for reading: " + latest.string());
    }

    // Read and verify header
    int magic = 0;
    int version = 0;
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x50494E4E || version != 1) {
        throw std::runtime_error("Invalid checkpoint file format");
    }

    // Read architecture
    int num_layers = 0;
    int input_dim = 0;
    ifs.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    ifs.read(reinterpret_cast<char*>(&input_dim), sizeof(input_dim));

    // Verify architecture matches
    if (num_layers != network.num_layers() || input_dim != network.input_dim()) {
        throw std::runtime_error("Checkpoint architecture mismatch");
    }

    // Read and verify layer dimensions
    for (int i = 0; i < num_layers; ++i) {
        int out_features = 0;
        ifs.read(reinterpret_cast<char*>(&out_features), sizeof(out_features));
        if (out_features != network.layer(i).out_features()) {
            throw std::runtime_error("Checkpoint layer dimension mismatch");
        }
    }

    // Read and verify layer sizes, then load weights
    for (int i = 0; i < num_layers; ++i) {
        auto& layer = network.layer(i);
        
        int w_count = 0;
        int b_count = 0;
        ifs.read(reinterpret_cast<char*>(&w_count), sizeof(w_count));
        ifs.read(reinterpret_cast<char*>(layer.weight_data()), w_count * sizeof(nn::TYPE_VAL));
        
        ifs.read(reinterpret_cast<char*>(&b_count), sizeof(b_count));
        ifs.read(reinterpret_cast<char*>(layer.bias_data()), b_count * sizeof(nn::TYPE_VAL));
        
        // Verify sizes
        int expected_w = layer.in_features() * layer.out_features();
        int expected_b = layer.out_features();
        if (w_count != expected_w || b_count != expected_b) {
            throw std::runtime_error("Checkpoint layer size mismatch");
        }
    }

    ifs.close();
    Logger::instance().info("Loaded checkpoint from " + latest.string());
}

}  // namespace pinn::utils
