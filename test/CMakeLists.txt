message(STATUS "Configuring tests")

set(TEST_TARGETS
    gelu_forward
    residual_forward
    layernorm_forward
    softmax_forward
    attention_forward
)

foreach(TARGET_NAME ${TEST_TARGETS})
    message(STATUS "Configuring test for ${TARGET_NAME}")
    add_executable(${TARGET_NAME}_test
        ${TARGET_NAME}_test.cu
        ${CMAKE_SOURCE_DIR}/src/project/${TARGET_NAME}.cu
        ${CMAKE_SOURCE_DIR}/src/reference/reference.cpp
        ${CMAKE_SOURCE_DIR}/src/common/rng.cpp
        ${CMAKE_SOURCE_DIR}/src/common/check_tensor.cpp
    )
    target_link_libraries(${TARGET_NAME}_test PRIVATE GTest::gtest GTest::gtest_main)
    gtest_discover_tests(${TARGET_NAME}_test
        PROPERTIES ENVIRONMENT "GTEST_COLOR=1"
    )
endforeach()


