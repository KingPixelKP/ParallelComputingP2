#ifndef COMPAR2627_TRANSFORMER_LINEAR_FORWARD_H
#define COMPAR2627_TRANSFORMER_LINEAR_FORWARD_H

// Use in the assembly of the kernels into a processing pipeline
// Already implemented in assemble/linear_forward.cpp

#include <cublas_v2.h>

void linear_forward_cublas(cublasHandle_t handle, float* d_y, const float* d_x,
                            const float* d_W, int rows, int in_features, int out_features);

void linear_forward_cpu(float* y, const float* x, const float* W,
                                       int rows, int in_features, int out_features);
#endif // COMPAR2627_TRANSFORMER_LINEAR_FORWARD_H
