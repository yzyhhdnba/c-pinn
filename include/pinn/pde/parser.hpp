#pragma once

#include <memory>
#include <string>

#include "pinn/pde/pde.hpp"

namespace pinn::pde {

class PdeParser {
  public:
    explicit PdeParser(std::string expression);

    PdeFunction build() const;

  private:
    std::string expression_;
};

}  // namespace pinn::pde
