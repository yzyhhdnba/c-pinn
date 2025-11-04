#pragma once

#include <memory>
#include <vector>

namespace pinn::utils {

class CallbackContext {
  public:
    int epoch{0};
    double loss{0.0};
};

class Callback {
  public:
    virtual ~Callback() = default;
    virtual void on_epoch_begin(const CallbackContext& /*ctx*/) {}
    virtual void on_epoch_end(const CallbackContext& /*ctx*/) {}
};

class CallbackRegistry {
  public:
    void add(std::shared_ptr<Callback> callback);
    void epoch_begin(const CallbackContext& ctx);
    void epoch_end(const CallbackContext& ctx);

  private:
    std::vector<std::shared_ptr<Callback>> callbacks_;
};

}  // namespace pinn::utils
