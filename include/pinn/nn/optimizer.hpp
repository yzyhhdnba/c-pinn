#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pinn/nn/fnn.hpp"
#include "pinn/types.hpp"

namespace pinn::nn {

// Pure C optimizer abstraction for FNN training
class Optimizer {
  public:
    virtual ~Optimizer() = default;

    // Perform one optimization step
    virtual void step() = 0;

    // Zero out accumulated gradients
    virtual void zero_grad() = 0;

    // Update learning rate
    virtual void set_learning_rate(Scalar lr) = 0;

    // Get current learning rate
    virtual Scalar get_learning_rate() const = 0;
};

class AdamOptimizer : public Optimizer {
  public:
    struct Options {
        Scalar lr{1e-3};
        Scalar beta1{0.9};
        Scalar beta2{0.999};
        Scalar epsilon{1e-8};
        Scalar weight_decay{0.0};
    };

    AdamOptimizer(Fnn& network, Options options);
    ~AdamOptimizer() override;

    void step() override;
    void zero_grad() override;
    void set_learning_rate(Scalar lr) override;
    Scalar get_learning_rate() const override;

  private:
    Fnn& network_;
    Options options_;
    int t_{0};

    // Adam state (flat arrays)
    TYPE_VAL* m_w_{nullptr};
    TYPE_VAL* v_w_{nullptr};
    TYPE_VAL* m_b_{nullptr};
    TYPE_VAL* v_b_{nullptr};

    // Gradient buffers
    TYPE_VAL* grads_w_{nullptr};
    TYPE_VAL* grads_b_{nullptr};

    int num_layers_{0};
    std::vector<int> layer_dims_;
};

class LbfgsOptimizer : public Optimizer {
  public:
    struct Options {
        Scalar lr{1.0};
        int max_iter{20};
        int history_size{10};
        Scalar tolerance_grad{1e-7};
        int line_search_max_evaluations{20};
    };

    LbfgsOptimizer(Fnn& network, Options options);
    ~LbfgsOptimizer() override;

    void step() override;
    void zero_grad() override;
    void set_learning_rate(Scalar lr) override;
    Scalar get_learning_rate() const override;

  private:
    Fnn& network_;
    Options options_;
};

// Gradient clipping utility
void clip_grad_norm_(Fnn& network, Scalar max_norm);

}  // namespace pinn::nn
