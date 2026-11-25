
#include <iostream>
#include "project.h"

__global__ static void gelu_forward_kernel(float* out, const float* inp, int N) {

}

void gelu_forward(const int version, float* d_out, const float* d_inp, const int N) {
    const int block = 256; // TODO: test with other values
    const int grid = (N + block - 1) / block;

    switch (version) {
        case 1:
            gelu_forward_kernel<<<grid, block>>>(d_out, d_inp, N);
            break;

        default:
            std::cerr << "Unknown version number " << version << std::endl;
            std::exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}
