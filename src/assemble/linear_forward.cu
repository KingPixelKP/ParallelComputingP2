// Given: linear layer y = x @ W^T, via cuBLAS SGEMM (no bias here --
// add it with a trivial broadcast-add kernel if/when you wire this in,
// left as a small exercise since it isn't the pedagogical point of this
// project). Not part of the student optimization work -- provided so
// the assembled forward pass (assemble/block_forward.cu) has a
// correct, fast building block for every matmul that isn't
// attention/softmax/layernorm.
//
// UNVERIFIED in this form: there is no CUDA toolkit or cuBLAS available
// in the environment this starter kit was written in, so this has not
// been compiled or run. Sanity-check it on the lab machine before
// trusting it -- see the NOTE in cublas_linear.h.

#include <cstdlib>
#include <iostream>

#include <linear_forward.h>

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error " << __FILE__ << ":" << __LINE__ \
                  << ": status " << static_cast<int>(st) << std::endl; \
        std::exit(1); \
    } \
} while (0)

void linear_forward_cublas(cublasHandle_t handle, float* d_y, const float* d_x,
                            const float* d_W, int rows, int in_features, int out_features) {
    const float alpha = 1.0f, beta = 0.0f;
    // cuBLAS is column-major. Row-major y[rows,out] = x[rows,in] @ W[out,in]^T
    // is equivalent to the column-major product
    //   y_colmajor[out,rows] = W_colmajor[in,out]^T-as-stored * x_colmajor[in,rows]
    // and a row-major [out,in] buffer IS a column-major [in,out] buffer
    // with the same bytes -- so W can be passed directly (opA = T to
    // undo that reinterpretation) with no physical transpose needed.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              out_features, rows, in_features,
                              &alpha,
                              d_W, in_features,
                              d_x, in_features,
                              &beta,
                              d_y, out_features));
}


void linear_forward_cpu(float* y, const float* x, const float* W,
                                       int rows, int in_features, int out_features) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * in_features;
        float* yr = y + (size_t)r * out_features;
        for (int o = 0; o < out_features; o++) {
            const float* wrow = W + (size_t)o * in_features;
            float acc = 0.0f;
            for (int i = 0; i < in_features; i++) acc += xr[i] * wrow[i];
            yr[o] = acc;
        }
    }
}