cmake_minimum_required(VERSION 3.22)
project(pinn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PINN_BUILD_EXAMPLES "Build example executables" ON)
option(PINN_BUILD_TESTS "Build unit tests" ON)

set(_NVTX_STUB "/home/yzy/libtorch/lib/libnvToolsExt-847d78f2.so.1")
if(NOT TARGET CUDA::nvToolsExt AND EXISTS "${_NVTX_STUB}")
    add_library(CUDA::nvToolsExt SHARED IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES
        IMPORTED_LOCATION "${_NVTX_STUB}"
    )
endif()

find_package(Torch REQUIRED)

set(_TORCH_ALL_LIBS ${TORCH_LIBRARIES})
if(NOT _TORCH_ALL_LIBS AND TORCH_LIBRARY)
    list(APPEND _TORCH_ALL_LIBS ${TORCH_LIBRARY})
endif()
if(TORCH_LIBRARY)
    get_filename_component(_TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
    if(_TORCH_LIB_DIR)
        find_library(TORCH_CUDA_LIBRARY torch_cuda HINTS "${_TORCH_LIB_DIR}")
        if(TORCH_CUDA_LIBRARY)
            message(STATUS "Found torch_cuda library: ${TORCH_CUDA_LIBRARY}")
            list(APPEND _TORCH_ALL_LIBS ${TORCH_CUDA_LIBRARY})
        endif()
        if(EXISTS "${_TORCH_LIB_DIR}/libc10_cuda.so")
            set(TORCH_C10_CUDA_LIBRARY "${_TORCH_LIB_DIR}/libc10_cuda.so")
            list(APPEND _TORCH_ALL_LIBS ${TORCH_C10_CUDA_LIBRARY})
        endif()
    endif()
endif()

if(NOT TARGET Torch::Torch)
    add_library(Torch::Torch INTERFACE IMPORTED)
    if(_TORCH_ALL_LIBS)
        target_link_libraries(Torch::Torch INTERFACE ${_TORCH_ALL_LIBS})
    endif()
    if(TORCH_INCLUDE_DIRS)
        target_include_directories(Torch::Torch INTERFACE ${TORCH_INCLUDE_DIRS})
    endif()
    if(TORCH_CXX_FLAGS)
        separate_arguments(_TORCH_COMPILE_OPTIONS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        if(_TORCH_COMPILE_OPTIONS)
            target_compile_options(Torch::Torch INTERFACE ${_TORCH_COMPILE_OPTIONS})
        endif()
    endif()
else()
    if(TORCH_CUDA_LIBRARY)
        target_link_libraries(Torch::Torch INTERFACE ${TORCH_CUDA_LIBRARY})
    endif()
    if(TORCH_C10_CUDA_LIBRARY)
        target_link_libraries(Torch::Torch INTERFACE ${TORCH_C10_CUDA_LIBRARY})
    endif()
endif()

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(pinn
    src/geometry/interval.cpp
    src/geometry/rectangle.cpp
    src/geometry/sampling.cpp
    src/pde/pde.cpp
    src/pde/bc.cpp
    src/pde/parser.cpp
    src/nn/fnn.cpp
    src/nn/activation.cpp
    src/nn/initialization.cpp
    src/loss/loss_terms.cpp
    src/model/model.cpp
    src/model/trainer.cpp
    src/utils/config.cpp
    src/utils/callbacks.cpp
    src/utils/checkpoint.cpp
    src/utils/logger.cpp
)

target_include_directories(pinn
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(pinn
    PUBLIC
        Torch::Torch
        nlohmann_json::nlohmann_json
        Eigen3::Eigen
)

if(TORCH_CUDA_LIBRARY)
    target_link_libraries(pinn PUBLIC ${TORCH_CUDA_LIBRARY})
    if(TORCH_CUDA_LIBRARY)
        target_link_options(pinn PRIVATE "-Wl,--no-as-needed")
        target_link_libraries(pinn PUBLIC ${TORCH_CUDA_LIBRARY})
        if(TORCH_C10_CUDA_LIBRARY)
            target_link_libraries(pinn PUBLIC ${TORCH_C10_CUDA_LIBRARY})
        endif()
    endif()
    add_subdirectory(examples)
endif()

if(PINN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
