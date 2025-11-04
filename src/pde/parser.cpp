#include "pinn/pde/parser.hpp"

#include <stdexcept>

#include <torch/torch.h>

namespace pinn::pde {

PdeParser::PdeParser(std::string expression) : expression_{std::move(expression)} {}

PdeFunction PdeParser::build() const {
    const std::string expr = expression_;
    return [expr](const DifferentialData&) {
        throw std::runtime_error{"PDE parser evaluation not yet implemented for expression: " + expr};
        return torch::zeros({1}, torch::dtype(torch::kDouble));
    };
}

}  // namespace pinn::pde
