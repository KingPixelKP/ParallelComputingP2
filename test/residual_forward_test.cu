

#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>

#include "project.h"
#include "rng.h"
#include "reference.h"

void residual_forward_test(const int version) {
    int N = 1 << 20;
    srand(1234);

    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    float* h_ref = (float*)malloc(N * sizeof(float));
    fill_random(h_a, N, -2.0f, 2.0f);
    fill_random(h_b, N, -2.0f, 2.0f);
    residual_forward_cpu(h_ref, h_a, h_b, N);

    float *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    residual_forward(version, d_out, d_a, d_b, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(check_tensor(h_out, h_ref, N, 1e-4f, "residual_forward"));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int reps = 100;
    cudaEventRecord(start);
    for (int i = 0; i < reps; i++) residual_forward(version, d_out, d_a, d_b, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("version %d: %.4f ms/launch (avg over %d)\n", version, ms / reps, reps);

    free(h_a); free(h_b); free(h_out); free(h_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
}


TEST(ResidualForward, V1) {
    auto version = 1;
    residual_forward_test(version);
}

