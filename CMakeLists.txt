cmake_minimum_required(VERSION 3.22)
project(pinn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PINN_BUILD_EXAMPLES "Build example executables" ON)
option(PINN_BUILD_TESTS "Build unit tests" ON)

option(PINN_USE_TORCH "Build with LibTorch backend" OFF)

if(PINN_USE_TORCH)
    find_package(Torch REQUIRED)
    if(TORCH_CUDA_AVAILABLE)
        message(STATUS ">> LibTorch CUDA Support: ENABLED")
        add_definitions(-DPINN_USE_CUDA)
    else()
        message(STATUS ">> LibTorch CUDA Support: NOT FOUND (CPU Only)")
    endif()
endif()

find_package(nlohmann_json REQUIRED)

add_library(pinn
    src/core/tensor.cpp
    src/geometry/interval.cpp
    src/geometry/rectangle.cpp
    src/geometry/difference.cpp
    src/geometry/sampling.cpp
    src/pde/pde.cpp
    src/pde/parser.cpp
    src/nn/gemm.cpp
    src/nn/relu.cpp
    src/nn/tanh.cpp
    src/nn/sin.cpp
    src/nn/adam.cpp
    src/nn/flatten.cpp
    src/nn/lbfgs.cpp
    src/nn/activation.cpp
    src/nn/initialization.cpp
    src/nn/optimizer.cpp
    src/nn/fnn.cpp
    src/utils/config.cpp
    src/utils/callbacks.cpp
    src/utils/checkpoint_c.cpp
    src/utils/logger.cpp
)

if(PINN_USE_TORCH)
    target_sources(pinn PRIVATE
        src/pde/bc.cpp
        src/loss/loss_terms.cpp
        src/model/model.cpp
        src/model/trainer.cpp
        src/utils/checkpoint.cpp
        src/utils/visualization_callback.cpp
    )
endif()

target_include_directories(pinn
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pinn
    PUBLIC
        nlohmann_json::nlohmann_json
)

if(PINN_USE_TORCH)
    target_link_libraries(pinn PUBLIC Torch::Torch)
endif()

if(PINN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(PINN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
