
#include "project.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void residual_forward_kernel(float *out, const float *a,
                                        const float *b, int N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < N) {
    out[index] = a[index] + b[index];
  }
}

void residual_forward(int version, float *d_out, const float *d_a,
                      const float *d_b, int N) {
  const int block = 256; // TODO: test with other values
  const int grid = (N + block - 1) / block;

  switch (version) {
  case 1:
    residual_forward_kernel<<<grid, block>>>(d_out, d_a, d_b, N);
    break;

  default:
    std::cerr << "Unknown version number " << version << std::endl;
    std::exit(1);
  }
  CUDA_CHECK(cudaGetLastError());
}
