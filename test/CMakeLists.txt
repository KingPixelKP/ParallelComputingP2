enable_testing()

set(TEST_TARGETS
        gelu_forward
        residual_forward
        layernorm_forward
        softmax_forward
        attention_forward
)

foreach(TARGET_NAME ${TEST_TARGETS})
    add_executable(${TARGET_NAME}_test
            ${TARGET_NAME}_test.cu
            ${CMAKE_SOURCE_DIR}/src/project/${TARGET_NAME}.cu
            ${CMAKE_SOURCE_DIR}/src/reference/reference.cpp
            ${CMAKE_SOURCE_DIR}/src/common/rng.cpp
            ${CMAKE_SOURCE_DIR}/src/common/check_tensor.cpp
            )
    target_link_libraries(${TARGET_NAME}_test PRIVATE ${GTEST_BOTH_LIBRARIES})
endforeach()


