//
// Created by herve on 24/11/26.
//

#ifndef COMPAR2627_TRANSFORMER_CUDA_H
#define COMPAR2627_TRANSFORMER_CUDA_H

#include <cstdlib>
#include <iostream>

#ifdef CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        std::exit(1); \
    } \
} while (0)
#else
#define CUDA_CHECK(call) (call)
#endif

// Returns 1 (pass) if the max abs diff between actual/expected is within
// tol, else 0 (fail). Always prints a one-line summary -- use this from
// every kernel test harness so output format is consistent across the
// project.
int check_tensor(const float* actual, const float* expected, int n,
                                float tol, const char* name);


// Host launcher functions for each op's CUDA project. Each .cu file in
// this directory defines its own __global__ project (kernel1 = naive,
// given; kernel2 = TODO, yours) plus one of these launcher functions
// that dispatches on `version`. All pointers are device pointers.
//
// Build each file standalone (with its own main() test harness) via
// `make tests` (adds -DSTANDALONE_TEST), or link them together without
// a main() into the assembled pipeline via `make assemble` -- see the
// Makefile and assemble/block_forward.cu.

void gelu_forward(int version, float* d_out, const float* d_inp, int N);

void residual_forward(int version, float* d_out, const float* d_a, const float* d_b, int N);

void layernorm_forward(int version, float* d_out, const float* d_inp,
                        const float* d_weight, const float* d_bias,
                        int rows, int C, float eps);

void softmax_forward(int version, float* d_out, const float* d_inp, int rows, int C);

// q, k, v, d_out: device pointers, layout [B, T, NH, HS] contiguous
// (== [B, T, C] with C = NH*HS). T must not exceed MAX_T in
// attention_forward.cu (bump it there if you need a longer sequence).
void attention_forward(int version, float* d_out, const float* d_q, const float* d_k,
                        const float* d_v, int B, int T, int NH, int HS);

#endif //COMPAR2627_TRANSFORMER_CUDA_H
