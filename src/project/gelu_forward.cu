
#include "project.h"
#include <cuda_runtime.h>
#include <iostream>

constexpr float sqrt_2_over_pi = 0.7978845608028654f;

__global__ static void gelu_forward_kernel(float *out, const float *inp,
                                           int N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < N) {
    float x = inp[index];
    float cube = 0.044715f * x * x * x;
    out[index] = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + cube)));
  }
}

void gelu_forward(const int version, float *d_out, const float *d_inp,
                  const int N) {
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
