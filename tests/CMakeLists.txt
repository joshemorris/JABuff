# Helper macro to add a test easily
macro(add_jabuff_test TEST_NAME FILE_NAME)
    add_executable(${TEST_NAME} ${FILE_NAME})
    target_link_libraries(${TEST_NAME} PRIVATE JABuff)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endmacro()

# Register tests
add_jabuff_test(Test2D test_2d.cpp)
add_jabuff_test(Test3D test_3d.cpp)
add_jabuff_test(TestOLA test_ola.cpp)
add_jabuff_test(TestExceptions test_exceptions.cpp)
