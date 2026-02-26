add_executable(unit_tests
    tensor_test.cpp
)

target_link_libraries(unit_tests PRIVATE pinn)

add_test(NAME unit_tests COMMAND unit_tests)
