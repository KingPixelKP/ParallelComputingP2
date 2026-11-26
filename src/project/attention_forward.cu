// Attention forward: causal multi-head scaled-dot-product self-attention
// core (GPT-2 style), operating on already-projected Q/K/V (the QKV
// projection itself is a cuBLAS linear layer -- see
// common/cublas_linear.cu -- not part of this kernel).
//
// q, k, v, out: [B, T, NH, HS] contiguous (== [B, T, C] with C = NH*HS).
//
// This is the centerpiece kernel of the project

#include "project.h"
#include <iostream>

#define MAX_T                                                                  \
  256 // must be >= the T used at launch; bump if you need a longer sequence

__global__ void attention_forward_kernel_v1(float *out, const float *q,
                                            const float *k, const float *v,
                                            int B, int T, int NH, int HS,
                                            float *scores_buffer) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  int C = NH * HS;
  float scale = 1.0f / std::sqrt(static_cast<float>(HS));

  int b = ii / (T * NH);
  int t = (ii / NH) % T;
  int h = ii % NH;

  float *scores =
      scores_buffer + (size_t)ii * T; // Cast to size_t to avoid overflows

  if (b < B && t < T && h < NH) {
    const float *qt = q + ((size_t)b * T + t) * C + (size_t)h * HS;

    float maxv = -1e30f;
    for (int t2 = 0; t2 <= t; t2++) {
      const float *kt2 = k + ((size_t)b * T + t2) * C + (size_t)h * HS;
      float dot = 0.0f;
      for (int s = 0; s < HS; s++)
        dot += qt[s] * kt2[s];
      dot *= scale;
      scores[t2] = dot;
      if (dot > maxv)
        maxv = dot;
    }
    float sum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
      float e = std::exp(scores[t2] - maxv);
      scores[t2] = e;
      sum += e;
    }
    for (int t2 = 0; t2 <= t; t2++)
      scores[t2] /= sum;

    float *ot = out + ((size_t)b * T + t) * C + (size_t)h * HS;
    for (int s = 0; s < HS; s++)
      ot[s] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
      const float *vt2 = v + ((size_t)b * T + t2) * C + (size_t)h * HS;
      float w = scores[t2];
      for (int s = 0; s < HS; s++)
        ot[s] += w * vt2[s];
    }
  }
}

void attention_forward(int version, float *d_out, const float *d_q,
                       const float *d_k, const float *d_v, int B, int T, int NH,
                       int HS) {
  if (T > MAX_T) {
    std::cerr << "attention_forward: T=" << T << " exceeds MAX_T=" << MAX_T
              << ", bump MAX_T in attention_forward.cu" << std::endl;
    std::exit(1);
  }
  int total = B * NH * T;
  int block = 128; // TODO: test with other values
  int grid = (total + block - 1) / block;

  float *d_scores_buffer;
  CUDA_CHECK(cudaMalloc(&d_scores_buffer, B * NH * T * T * sizeof(float)));

  switch (version) {
  case 1:
    attention_forward_kernel_v1<<<grid, block>>>(d_out, d_q, d_k, d_v, B, T, NH,
                                                 HS, d_scores_buffer);
    break;

  default:
    std::cerr << "Unknown version number " << version << std::endl;
    std::exit(1);
  }

  CUDA_CHECK(cudaFree(d_scores_buffer));

  CUDA_CHECK(cudaGetLastError());
}
