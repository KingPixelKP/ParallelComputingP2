#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>

#include "project.h"
#include "rng.h"
#include "reference.h"

void attention_forward_test(const int version) {
    int B = 8, T = 256, NH = 4, HS = 64; // C = 256, matches the assignment's reference model
    int C = NH * HS;
    int N = B * T * C;
    srand(1234);

    float* h_q = (float*)malloc((size_t)N * sizeof(float));
    float* h_k = (float*)malloc((size_t)N * sizeof(float));
    float* h_v = (float*)malloc((size_t)N * sizeof(float));
    float* h_out = (float*)malloc((size_t)N * sizeof(float));
    float* h_ref = (float*)malloc((size_t)N * sizeof(float));
    fill_random(h_q, N, -1.0f, 1.0f);
    fill_random(h_k, N, -1.0f, 1.0f);
    fill_random(h_v, N, -1.0f, 1.0f);
    attention_forward_cpu(h_ref, h_q, h_k, h_v, B, T, NH, HS);

    float *d_q, *d_k, *d_v, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    attention_forward(version, d_out, d_q, d_k, d_v, B, T, NH, HS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(check_tensor(h_out, h_ref, N, 1e-3f, "attention_forward"));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int reps = 20;
    cudaEventRecord(start);
    for (int i = 0; i < reps; i++)
        attention_forward(version, d_out, d_q, d_k, d_v, B, T, NH, HS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("version %d: %.4f ms/launch (avg over %d)\n", version, ms / reps, reps);

    free(h_q); free(h_k); free(h_v); free(h_out); free(h_ref);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
}


TEST(AttentionForward, V1) {
    auto version = 1;
    attention_forward_test(version);
}
