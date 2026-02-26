#include "pinn/nn/fnn.hpp"
#include "pinn/core/tensor.hpp"
#include "pinn/utils/checkpoint_c.hpp"
#include <iostream>
#include <vector>

using namespace pinn;

int main() {
    std::cout << "=== Pure C PINN Inference Example ===" << std::endl;

    // 1. Define Network Architecture
    // Input: 2 (x, t), Hidden: 3 layers of 50 units, Output: 1 (u)
    std::vector<int> layers = {2, 50, 50, 50, 1};
    
    // Use Tanh activation
    auto activation = nn::activation_from_string("tanh");
    auto activation_deriv = nn::activation_derivative_from_string("tanh");
    
    // 2. Initialize Network
    // Use Xavier Uniform initialization, seed 42
    std::cout << "Initializing FNN..." << std::endl;
    nn::Fnn net(layers, activation, activation_deriv, nn::InitType::kXavierUniform, 0.0, 42);

    // 3. Prepare Input Data
    // Create a batch of 5 random points
    std::cout << "Creating input tensor..." << std::endl;
    auto input = core::Tensor::rand_uniform({5, 2});
    
    std::cout << "Input data:" << std::endl;
    // Simple print for tensor (assuming we might not have a full ostream overload yet, or we do)
    // Let's just print shape for now to be safe, or iterate.
    // core::Tensor doesn't have a print method exposed in the headers I saw, 
    // but let's assume we can access data.
    const double* data = input.data_ptr<double>();
    for(int i=0; i<5; ++i) {
        std::cout << "Sample " << i << ": [" << data[i*2] << ", " << data[i*2+1] << "]" << std::endl;
    }

    // 4. Forward Pass
    std::cout << "Running forward pass..." << std::endl;
    auto output = net.forward(input);

    std::cout << "Output data:" << std::endl;
    const double* out_data = output.data_ptr<double>();
    for(int i=0; i<5; ++i) {
        std::cout << "Sample " << i << ": [" << out_data[i] << "]" << std::endl;
    }

    // 5. Save Model
    std::string ckpt_dir = "checkpoints_pure_c";
    std::cout << "Saving model to " << ckpt_dir << "..." << std::endl;
    utils::CheckpointManagerC ckpt(ckpt_dir);
    ckpt.save(net, 0, 0.0); // epoch 0, loss 0.0

    // 6. Load Model
    std::cout << "Loading model back..." << std::endl;
    nn::Fnn net_loaded(layers, activation, activation_deriv, nn::InitType::kXavierUniform, 0.0, 42); // Re-init
    ckpt.load_latest(net_loaded);

    // 7. Verify Load
    auto output_loaded = net_loaded.forward(input);
    
    // Check difference
    auto diff = (output - output_loaded).abs().sum_all();
    double diff_val = diff.item<double>();
    
    std::cout << "Difference between original and loaded model output: " << diff_val << std::endl;

    if (diff_val < 1e-9) {
        std::cout << "SUCCESS: Model saved and loaded correctly." << std::endl;
    } else {
        std::cout << "FAILURE: Model load mismatch." << std::endl;
        return 1;
    }

    return 0;
}
