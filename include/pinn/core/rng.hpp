#pragma once

#include <cstdint>
#include <cmath>

namespace pinn::core {

// Simple C-style RNG (splitmix64 + Box-Muller) for deterministic results.
class Rng {
  public:
    explicit Rng(uint64_t seed = 0x9e3779b97f4a7c15ULL) : state_{seed} {}

    uint64_t next_u64() {
        // splitmix64
        uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    // Uniform in [0,1)
    double uniform01() {
        // Use top 53 bits
        const uint64_t x = next_u64();
        return (x >> 11) * (1.0 / 9007199254740992.0);
    }

    // Standard normal via Box-Muller
    double normal01() {
        // Avoid log(0)
        double u1 = uniform01();
        if (u1 < 1e-12) {
            u1 = 1e-12;
        }
        const double u2 = uniform01();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = 6.2831853071795864769 * u2;
        return r * std::cos(theta);
    }

    int64_t randint(int64_t low, int64_t high) {
        if (high <= low) {
            return low;
        }
        const uint64_t span = static_cast<uint64_t>(high - low);
        const uint64_t x = next_u64();
        return low + static_cast<int64_t>(x % span);
    }

  private:
    uint64_t state_;
};

}  // namespace pinn::core
