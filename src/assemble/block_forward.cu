// Pipeline for one pre-LN transformer block, GPT-2/llm.c style
// (causal self-attention, not a bidirectional/BERT-style encoder):
//
//   LN1 -> QKV projection (cuBLAS) -> attention -> output projection (cuBLAS)
//       -> residual add -> LN2 -> FFN (linear -> GELU -> linear, cuBLAS + gelu kernel)
//       -> residual add
//
// This links against the kernel launcher functions declared in
// project.h. Once you've implemented and validated your optimized kernels
// individually, swap them in here by changing the version argument.
//
// Usage: block_forward [version] [gpu|cpu]
//   version  kernel version passed to every launcher (default 1)
//   gpu      (default) time the GPU forward pass, then check its output
//            against the CPU reference chain (the check is not timed)
//   cpu      time the CPU reference chain only (the baseline for speedups)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include "project.h"
#include "reference.h"
#include "rng.h"
#include "linear_forward.h"

// Model sizes (the assignment's reference model).
static constexpr int B = 8, T = 128, C = 256, NH = 4, HS = C / NH, FFN = 4 * C;
static constexpr int ROWS = B * T;
static constexpr float EPS = 1e-5f;

// Number of timed GPU forward passes (after one untimed warm-up pass,
// which absorbs one-off costs such as cuBLAS initialization).
static constexpr int GPU_REPS = 10;

// Input and weights, shared by the GPU and CPU pipelines.
struct Params {
    float *x;
    float *ln1_w, *ln1_b, *qkv_w, *attn_proj_w;
    float *ln2_w, *ln2_b, *ffn1_w, *ffn2_w;
};

static float *to_device(const float *h, size_t n) {
    float *d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

static float *device_alloc(size_t n) {
    float *d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    return d;
}

static float *host_alloc(size_t n) {
    return (float *) malloc(n * sizeof(float));
}

// ----------------------------------------------------------------------
// Random, untrained input + weights. This checks the pipeline *computes
// the right thing*, not that it produces meaningful text; the same
// values are used on the GPU path and the CPU reference chain so the two
// are directly comparable.
// ----------------------------------------------------------------------
static Params make_host_params() {
    Params h;
    h.x = host_alloc((size_t) ROWS * C);
    h.ln1_w = host_alloc(C);
    h.ln1_b = host_alloc(C);
    h.qkv_w = host_alloc((size_t) 3 * C * C);
    h.attn_proj_w = host_alloc((size_t) C * C);
    h.ln2_w = host_alloc(C);
    h.ln2_b = host_alloc(C);
    h.ffn1_w = host_alloc((size_t) FFN * C);
    h.ffn2_w = host_alloc((size_t) C * FFN);

    fill_random(h.x, ROWS * C, -1.0f, 1.0f);
    fill_random(h.ln1_w, C, 0.8f, 1.2f);
    fill_random(h.ln1_b, C, -0.1f, 0.1f);
    fill_random(h.qkv_w, 3 * C * C, -0.1f, 0.1f);
    fill_random(h.attn_proj_w, C * C, -0.1f, 0.1f);
    fill_random(h.ln2_w, C, 0.8f, 1.2f);
    fill_random(h.ln2_b, C, -0.1f, 0.1f);
    fill_random(h.ffn1_w, FFN * C, -0.1f, 0.1f);
    fill_random(h.ffn2_w, C * FFN, -0.1f, 0.1f);
    return h;
}

static void free_host_params(Params &h) {
    free(h.x);
    free(h.ln1_w);
    free(h.ln1_b);
    free(h.qkv_w);
    free(h.attn_proj_w);
    free(h.ln2_w);
    free(h.ln2_b);
    free(h.ffn1_w);
    free(h.ffn2_w);
}

// ----------------------------------------------------------------------
// GPU pipeline
// ----------------------------------------------------------------------

// Intermediate device buffers of the GPU forward pass.
struct GpuBuffers {
    float *ln1_out, *qkv, *q, *k, *v, *attn_out, *attn_proj_out;
    float *resid1, *ln2_out, *ffn1_out, *gelu_out, *ffn2_out, *resid2;
};

static GpuBuffers make_gpu_buffers() {
    GpuBuffers b;
    b.ln1_out = device_alloc((size_t) ROWS * C);
    b.qkv = device_alloc((size_t) ROWS * 3 * C);
    b.q = device_alloc((size_t) ROWS * C);
    b.k = device_alloc((size_t) ROWS * C);
    b.v = device_alloc((size_t) ROWS * C);
    b.attn_out = device_alloc((size_t) ROWS * C);
    b.attn_proj_out = device_alloc((size_t) ROWS * C);
    b.resid1 = device_alloc((size_t) ROWS * C);
    b.ln2_out = device_alloc((size_t) ROWS * C);
    b.ffn1_out = device_alloc((size_t) ROWS * FFN);
    b.gelu_out = device_alloc((size_t) ROWS * FFN);
    b.ffn2_out = device_alloc((size_t) ROWS * C);
    b.resid2 = device_alloc((size_t) ROWS * C);
    return b;
}

static void free_gpu_buffers(GpuBuffers &b) {
    float *all[] = {
        b.ln1_out, b.qkv, b.q, b.k, b.v, b.attn_out, b.attn_proj_out,
        b.resid1, b.ln2_out, b.ffn1_out, b.gelu_out, b.ffn2_out, b.resid2
    };
    for (float *p: all) CUDA_CHECK(cudaFree(p));
}

// One forward pass on the GPU. Input and weights in `d` (device
// pointers); result left in b.resid2. Does not end with a device
// synchronization: callers must synchronize before reading the result.
static void gpu_forward(int version, cublasHandle_t cublas, const Params &d, const GpuBuffers &b) {
    layernorm_forward(version, b.ln1_out, d.x, d.ln1_w, d.ln1_b, ROWS, C, EPS);
    linear_forward_cublas(cublas, b.qkv, b.ln1_out, d.qkv_w, ROWS, C, 3 * C);

    // b.qkv is [ROWS, 3*C] (Q, K, V concatenated per row); attention_forward
    // expects each of Q/K/V as its own contiguous [ROWS, C] buffer, so pull
    // the three C-wide column slices out with a strided 2D device-to-device
    // copy (src pitch 3*C floats/row, dst pitch C floats/row).
    CUDA_CHECK(cudaMemcpy2D(b.q, C * sizeof(float), b.qkv + 0 * C, 3 * (size_t) C * sizeof(float),
                            C * sizeof(float), ROWS, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(b.k, C * sizeof(float), b.qkv + 1 * C, 3 * (size_t) C * sizeof(float),
                            C * sizeof(float), ROWS, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(b.v, C * sizeof(float), b.qkv + 2 * C, 3 * (size_t) C * sizeof(float),
                            C * sizeof(float), ROWS, cudaMemcpyDeviceToDevice));

    attention_forward(version, b.attn_out, b.q, b.k, b.v, B, T, NH, HS);
    linear_forward_cublas(cublas, b.attn_proj_out, b.attn_out, d.attn_proj_w, ROWS, C, C);
    residual_forward(version, b.resid1, d.x, b.attn_proj_out, ROWS * C);
    layernorm_forward(version, b.ln2_out, b.resid1, d.ln2_w, d.ln2_b, ROWS, C, EPS);
    linear_forward_cublas(cublas, b.ffn1_out, b.ln2_out, d.ffn1_w, ROWS, C, FFN);
    gelu_forward(version, b.gelu_out, b.ffn1_out, ROWS * FFN);
    linear_forward_cublas(cublas, b.ffn2_out, b.gelu_out, d.ffn2_w, ROWS, FFN, C);
    residual_forward(version, b.resid2, b.resid1, b.ffn2_out, ROWS * C);
}

// Copies the inputs to the device, times the GPU forward pass and copies
// the result back into `out` (ROWS*C floats). The timed region is the
// forward pass only (device input -> device output): allocation, the
// host<->device transfers and the warm-up pass are excluded.
static void run_gpu(int version, const Params &h, float *out) {
    Params d;
    d.x = to_device(h.x, (size_t) ROWS * C);
    d.ln1_w = to_device(h.ln1_w, C);
    d.ln1_b = to_device(h.ln1_b, C);
    d.qkv_w = to_device(h.qkv_w, (size_t) 3 * C * C);
    d.attn_proj_w = to_device(h.attn_proj_w, (size_t) C * C);
    d.ln2_w = to_device(h.ln2_w, C);
    d.ln2_b = to_device(h.ln2_b, C);
    d.ffn1_w = to_device(h.ffn1_w, (size_t) FFN * C);
    d.ffn2_w = to_device(h.ffn2_w, (size_t) C * FFN);
    GpuBuffers b = make_gpu_buffers();

    cublasHandle_t cublas;
    cublasCreate(&cublas);

    // Warm-up (untimed).
    gpu_forward(version, cublas, d, b);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < GPU_REPS; i++) gpu_forward(version, cublas, d, b);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU (version %d): %.4f ms/forward pass (avg over %d)\n", version, ms / GPU_REPS, GPU_REPS);

    CUDA_CHECK(cudaMemcpy(out, b.resid2, (size_t) ROWS * C * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cublasDestroy(cublas);
    free_gpu_buffers(b);
    float *params[] = {
        d.x, d.ln1_w, d.ln1_b, d.qkv_w, d.attn_proj_w,
        d.ln2_w, d.ln2_b, d.ffn1_w, d.ffn2_w
    };
    for (float *p: params) CUDA_CHECK(cudaFree(p));
}

// ----------------------------------------------------------------------
// CPU reference chain: same ops, same order, same weights (reference.h),
// used both as the correctness oracle for the GPU pipeline and as the
// CPU baseline. Result written to `out` (ROWS*C floats).
// ----------------------------------------------------------------------
static void cpu_forward(const Params &h, float *out) {
    float *ln1_out = host_alloc((size_t) ROWS * C);
    float *qkv = host_alloc((size_t) ROWS * 3 * C);
    float *q = host_alloc((size_t) ROWS * C);
    float *k = host_alloc((size_t) ROWS * C);
    float *v = host_alloc((size_t) ROWS * C);
    float *attn_out = host_alloc((size_t) ROWS * C);
    float *attn_proj_out = host_alloc((size_t) ROWS * C);
    float *resid1 = host_alloc((size_t) ROWS * C);
    float *ln2_out = host_alloc((size_t) ROWS * C);
    float *ffn1_out = host_alloc((size_t) ROWS * FFN);
    float *gelu_out = host_alloc((size_t) ROWS * FFN);
    float *ffn2_out = host_alloc((size_t) ROWS * C);

    layernorm_forward_cpu(ln1_out, h.x, h.ln1_w, h.ln1_b, ROWS, C, EPS);
    linear_forward_cpu(qkv, ln1_out, h.qkv_w, ROWS, C, 3 * C);
    for (int r = 0; r < ROWS; r++) {
        std::copy_n(qkv + (size_t) r * 3 * C + 0 * C, C, q + (size_t) r * C);
        std::copy_n(qkv + (size_t) r * 3 * C + 1 * C, C, k + (size_t) r * C);
        std::copy_n(qkv + (size_t) r * 3 * C + 2 * C, C, v + (size_t) r * C);
    }
    attention_forward_cpu(attn_out, q, k, v, B, T, NH, HS);
    linear_forward_cpu(attn_proj_out, attn_out, h.attn_proj_w, ROWS, C, C);
    residual_forward_cpu(resid1, h.x, attn_proj_out, ROWS * C);
    layernorm_forward_cpu(ln2_out, resid1, h.ln2_w, h.ln2_b, ROWS, C, EPS);
    linear_forward_cpu(ffn1_out, ln2_out, h.ffn1_w, ROWS, C, FFN);
    gelu_forward_cpu(gelu_out, ffn1_out, ROWS * FFN);
    linear_forward_cpu(ffn2_out, gelu_out, h.ffn2_w, ROWS, FFN, C);
    residual_forward_cpu(out, resid1, ffn2_out, ROWS * C);

    float *all[] = {
        ln1_out, qkv, q, k, v, attn_out, attn_proj_out,
        resid1, ln2_out, ffn1_out, gelu_out, ffn2_out
    };
    for (float *p: all) free(p);
}

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog << " [version] [gpu|cpu]" << std::endl;
    std::exit(1);
}

int main(int argc, char **argv) {
    if (argc > 3) usage(argv[0]);
    const int version = (argc > 1) ? std::stoi(argv[1]) : 1;
    std::string device = (argc > 2) ? argv[2] : "gpu";
    std::transform(device.begin(), device.end(), device.begin(), ::tolower);
    if (device != "gpu" && device != "cpu") usage(argv[0]);

    std::srand(1234);
    Params h = make_host_params();
    float *ref_out = host_alloc((size_t) ROWS * C);
    int ok = 1;

    if (device == "cpu") {
        auto t0 = std::chrono::steady_clock::now();
        cpu_forward(h, ref_out);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("CPU: %.4f ms/forward pass\n", ms);
    }
    else {
        float *gpu_out = host_alloc((size_t) ROWS * C);
        run_gpu(version, h, gpu_out);

        // Correctness check against the CPU reference (untimed).
        // Looser tolerance than the per-kernel tests (1e-3/1e-4): floating-point
        // error compounds across ~10 chained ops here, so some growth in max
        // abs diff is expected and NOT itself a bug. If this fails by orders
        // of magnitude more than that, suspect the QKV split first.
        cpu_forward(h, ref_out);
        ok = check_tensor(gpu_out, ref_out, ROWS * C, 2e-2f, "block_forward (end-to-end)");
        free(gpu_out);
    }

    free(ref_out);
    free_host_params(h);
    return ok ? 0 : 1;
}
