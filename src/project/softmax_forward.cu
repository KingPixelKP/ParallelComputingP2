// Softmax forward: numerically-stable row-wise softmax. Sized in the
// standalone test below to match one attention head's score-matrix row
// shape (rows = B*NH*T, C = T) so the benchmark number is representative
// of what this op costs inside the actual attention computation.

#include <iostream>
#include "project.h"


__global__ void softmax_forward_kernel(float* out, const float* inp, int rows, int C) {

}

void softmax_forward(int version, float* d_out, const float* d_inp, int rows, int C) {
    switch (version) {
        case 1: {
            int block = 128;
            int grid = (rows + block - 1) / block;
            softmax_forward_kernel<<<grid, block>>>(d_out, d_inp, rows, C);
            break;
        }

        default:
            std::cerr << "Unknown version number " << version << std::endl;
            std::exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}
