// Attention forward: causal multi-head scaled-dot-product self-attention
// core (GPT-2 style), operating on already-projected Q/K/V (the QKV
// projection itself is a cuBLAS linear layer -- see
// common/cublas_linear.cu -- not part of this kernel).
//
// q, k, v, out: [B, T, NH, HS] contiguous (== [B, T, C] with C = NH*HS).
//
// This is the centerpiece kernel of the project

#include <iostream>
#include "project.h"

#define MAX_T 256  // must be >= the T used at launch; bump if you need a longer sequence

__global__ void attention_forward_kernel(float* out, const float* q, const float* k,
                                           const float* v, int B, int T, int NH, int HS) {
}


void attention_forward(int version, float* d_out, const float* d_q, const float* d_k,
                        const float* d_v, int B, int T, int NH, int HS) {
    if (T > MAX_T) {
        std::cerr << "attention_forward: T=" << T << " exceeds MAX_T=" << MAX_T
                  << ", bump MAX_T in attention_forward.cu" << std::endl;
        std::exit(1);
    }
    int total = B * NH * T;
    int block = 128;  // TODO: test with other values
    int grid = (total + block - 1) / block;

    switch (version) {
        case 1:
            attention_forward_kernel<<<grid, block>>>(d_out, d_q, d_k, d_v, B, T, NH, HS);
            break;

        default:
            std::cerr << "Unknown version number " << version << std::endl;
            std::exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

