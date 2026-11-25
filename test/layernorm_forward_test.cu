#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>

#include "project.h"
#include "rng.h"
#include "reference.h"

static void layernorm_forward_test(const int version) {
    int B = 8, T = 128, C = 256; // matches the assignment's reference model (Section 4)
    int rows = B * T;
    int N = rows * C;
    float eps = 1e-5f;
    srand(1234);

    float* h_inp = (float*)malloc((size_t)N * sizeof(float));
    float* h_w = (float*)malloc(C * sizeof(float));
    float* h_b = (float*)malloc(C * sizeof(float));
    float* h_out = (float*)malloc((size_t)N * sizeof(float));
    float* h_ref = (float*)malloc((size_t)N * sizeof(float));
    fill_random(h_inp, N, -2.0f, 2.0f);
    fill_random(h_w, C, 0.5f, 1.5f);
    fill_random(h_b, C, -0.5f, 0.5f);
    layernorm_forward_cpu(h_ref, h_inp, h_w, h_b, rows, C, eps);

    float *d_inp, *d_w, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_inp, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inp, h_inp, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, C * sizeof(float), cudaMemcpyHostToDevice));

    layernorm_forward(version, d_out, d_inp, d_w, d_b, rows, C, eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(check_tensor(h_out, h_ref, N, 1e-3f, "layernorm_forward"));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int reps = 100;
    cudaEventRecord(start);
    for (int i = 0; i < reps; i++) layernorm_forward(version, d_out, d_inp, d_w, d_b, rows, C, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("version %d: %.4f ms/launch (avg over %d)\n", version, ms / reps, reps);

    free(h_inp); free(h_w); free(h_b); free(h_out); free(h_ref);
    cudaFree(d_inp); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
}


TEST(LayernormForward, V1) {
    constexpr auto version = 1;
    layernorm_forward_test(version);
}
