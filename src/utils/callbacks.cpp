#include "pinn/utils/callback.hpp"

#include <utility>

namespace pinn::utils {

void CallbackRegistry::add(std::shared_ptr<Callback> callback) {
    callbacks_.push_back(std::move(callback));
}

void CallbackRegistry::epoch_begin(const CallbackContext& ctx) {
    for (auto& cb : callbacks_) {
        cb->on_epoch_begin(ctx);
    }
}

void CallbackRegistry::epoch_end(const CallbackContext& ctx) {
    for (auto& cb : callbacks_) {
        cb->on_epoch_end(ctx);
    }
}

}  // namespace pinn::utils
