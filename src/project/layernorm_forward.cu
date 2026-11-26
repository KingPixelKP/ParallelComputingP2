// LayerNorm forward: per-row (rows = B*T) over C channels, affine
// weight/bias.


#include <iostream>
#include "project.h"

__global__ void layernorm_forward_kernel(float* out, const float* inp,
                                           const float* weight, const float* bias,
                                           int rows, int C, float eps) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;

    if (ii < rows) {
        const float*x = inp + static_cast<size_t>(ii) * C;
        float* y = out + static_cast<size_t>(ii) * C;
        float mean = 0.0f;
        for (int c = 0; c < C; c++) mean += x[c];
        mean /= C;
        float var = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = x[c] - mean;
            var += d * d;
        }
        var /= C;
        float rstd = 1.0f / std::sqrt(var + eps);
        for (int c = 0; c < C; c++) {
            y[c] = (x[c] - mean) * rstd * weight[c] + bias[c];
        }
    }
}



void layernorm_forward(int version, float* d_out, const float* d_inp,
                        const float* d_weight, const float* d_bias,
                        int rows, int C, float eps) {
    switch (version) {
        case 1: {
            int block = 128;
            int grid = (rows + block - 1) / block;
            layernorm_forward_kernel<<<grid, block>>>(d_out, d_inp, d_weight, d_bias, rows, C, eps);
            break;
        }

        default:
            std::cerr << "Unknown version number " << version << std::endl;
            std::exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}
