cmake_minimum_required(VERSION 3.18)
project(compar-project2 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##############################################################
# Autodetect CUDA architecture(s)
##############################################################

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
message("-- CUDA Architecture(s): " ${CMAKE_CUDA_ARCHITECTURES})

find_package(CUDAToolkit REQUIRED)

##############################################################
# Project configuration
##############################################################

# Output all binaries into the `bin/` directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Common compiler flags and include directories
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

include_directories(include)

##############################################################
# Reference implementation
##############################################################
add_executable(reference
        src/reference/reference.cpp
        src/common/rng.cpp
        src/reference/dump_test_case.cpp)

if(UNIX)
    target_link_libraries(reference PRIVATE m)
endif()
set_target_properties(reference PROPERTIES EXCLUDE_FROM_ALL TRUE)


##############################################################
# Tests
##############################################################

add_subdirectory(third_party)
include_directories(${GTEST_HOME}/googletest/include)
add_subdirectory (test)

##############################################################
# Assemble
##############################################################

# Kernel source files
set(KERNEL_SRCS
        src/project/gelu_forward.cu
        src/project/residual_forward.cu
        src/project/layernorm_forward.cu
        src/project/softmax_forward.cu
        src/project/attention_forward.cu
)

add_executable(block_forward
        src/assemble/block_forward.cu
        src/assemble/linear_forward.cu
        src/reference/reference.cpp
        src/common/rng.cpp
        src/common/check_tensor.cpp
        ${KERNEL_SRCS}
)

if(UNIX)
    target_link_libraries(reference PRIVATE m)
endif()
target_link_libraries(block_forward PRIVATE CUDA::cublas)
set_target_properties(block_forward PROPERTIES EXCLUDE_FROM_ALL TRUE)
