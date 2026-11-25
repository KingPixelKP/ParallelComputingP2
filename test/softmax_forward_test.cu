

#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>

#include "project.h"
#include "rng.h"
#include "reference.h"

void softmax_forward_test(const int version) {
    int B = 8, T = 128, NH = 4; // matches the assignment's reference model (Section 4)
    int rows = B * NH * T;
    int C = T;
    int N = rows * C;
    srand(1234);

    float* h_inp = (float*)malloc((size_t)N * sizeof(float));
    float* h_out = (float*)malloc((size_t)N * sizeof(float));
    float* h_ref = (float*)malloc((size_t)N * sizeof(float));
    fill_random(h_inp, N, -3.0f, 3.0f);
    softmax_forward_cpu(h_ref, h_inp, rows, C);

    float *d_inp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_inp, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inp, h_inp, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    softmax_forward(version, d_out, d_inp, rows, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(check_tensor(h_out, h_ref, N, 1e-3f, "softmax_forward"));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int reps = 50;
    cudaEventRecord(start);
    for (int i = 0; i < reps; i++) softmax_forward(version, d_out, d_inp, rows, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("version %d: %.4f ms/launch (avg over %d)\n", version, ms / reps, reps);

    free(h_inp); free(h_out); free(h_ref);
    cudaFree(d_inp); cudaFree(d_out);
}


TEST(SoftmaxForward, V1) {
    auto version = 1;
    softmax_forward_test(version);
}

