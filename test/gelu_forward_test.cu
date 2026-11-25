

#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>

#include "project.h"
#include "rng.h"
#include "reference.h"

void gelue_test(const int version) {
    constexpr int N = 1 << 20; // 1M elements; big enough that timing isn't launch-overhead-dominated
    srand(1234);

    float* h_inp = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    float* h_ref = (float*)malloc(N * sizeof(float));
    fill_random(h_inp, N, -3.0f, 3.0f);
    gelu_forward_cpu(h_ref, h_inp, N);

    float *d_inp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_inp, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_inp, h_inp, N * sizeof(float), cudaMemcpyHostToDevice));

    gelu_forward(version, d_out, d_inp, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    check_tensor(h_out, h_ref, N, 1e-3f, "gelu_forward");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int reps = 100;
    cudaEventRecord(start);
    for (int i = 0; i < reps; i++) gelu_forward(version, d_out, d_inp, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("version %d: %.4f ms/launch (avg over %d)\n", version, ms / reps, reps);

    free(h_inp); free(h_out); free(h_ref);
    cudaFree(d_inp); cudaFree(d_out);
}


TEST(Gelu, V1) {
    auto version = 1;
    gelue_test(version);
}

