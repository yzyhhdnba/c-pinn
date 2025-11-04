add_executable(example_poisson
    poisson.cpp
)

target_link_libraries(example_poisson PRIVATE pinn)
if(TORCH_CUDA_LIBRARY)
    target_link_options(example_poisson PRIVATE "-Wl,--no-as-needed")
    target_link_libraries(example_poisson PRIVATE ${TORCH_CUDA_LIBRARY})
    if(TORCH_C10_CUDA_LIBRARY)
        target_link_libraries(example_poisson PRIVATE ${TORCH_C10_CUDA_LIBRARY})
    endif()
endif()

add_executable(example_advection
    advection.cpp
)

target_link_libraries(example_advection PRIVATE pinn)
if(TORCH_CUDA_LIBRARY)
    target_link_options(example_advection PRIVATE "-Wl,--no-as-needed")
    target_link_libraries(example_advection PRIVATE ${TORCH_CUDA_LIBRARY})
    if(TORCH_C10_CUDA_LIBRARY)
        target_link_libraries(example_advection PRIVATE ${TORCH_C10_CUDA_LIBRARY})
    endif()
endif()

add_executable(example_burgers
    burgers.cpp
)

target_link_libraries(example_burgers PRIVATE pinn)
if(TORCH_CUDA_LIBRARY)
    target_link_options(example_burgers PRIVATE "-Wl,--no-as-needed")
    target_link_libraries(example_burgers PRIVATE ${TORCH_CUDA_LIBRARY})
    if(TORCH_C10_CUDA_LIBRARY)
        target_link_libraries(example_burgers PRIVATE ${TORCH_C10_CUDA_LIBRARY})
    endif()
endif()
