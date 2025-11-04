#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

namespace pinn::utils {

class Logger {
  public:
    static Logger& instance();

    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);

  private:
    Logger() = default;
    std::mutex mutex_;
};

}  // namespace pinn::utils
