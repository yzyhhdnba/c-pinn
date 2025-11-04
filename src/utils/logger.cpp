#include "pinn/utils/logger.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>

namespace pinn::utils {

namespace {
std::string timestamp() {
    using clock = std::chrono::system_clock;
    auto now = clock::now();
    auto time = clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

void log_line(std::ostream& stream, const std::string& level, const std::string& message) {
    stream << "[" << timestamp() << "] [" << level << "] " << message << std::endl;
}

}  // namespace

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::info(const std::string& message) {
    std::lock_guard<std::mutex> lock{mutex_};
    log_line(std::cout, "INFO", message);
}

void Logger::warn(const std::string& message) {
    std::lock_guard<std::mutex> lock{mutex_};
    log_line(std::cout, "WARN", message);
}

void Logger::error(const std::string& message) {
    std::lock_guard<std::mutex> lock{mutex_};
    log_line(std::cerr, "ERROR", message);
}

}  // namespace pinn::utils
