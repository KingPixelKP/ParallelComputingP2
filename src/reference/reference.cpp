
#include <cmath>
#include <cstdlib>

// ---------------------------------------------------------------------
// CPU reference implementations used as the correctness oracle for the
// CUDA project in this project. Pure C/C++, no CUDA dependency, so these
// can be compiled standalone (gcc, see validate/dump_test_case.c) for
// independent validation, or included inside a .cu test harness to
// generate expected outputs for a GPU kernel's correctness check.
//
// Cross-checked against an independent numpy re-implementation in
// validate/verify_reference.py -- see that file for the comparison.
// ---------------------------------------------------------------------

// GELU, GPT-2's tanh approximation.
void gelu_forward_cpu(float* out, const float* inp, int N) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + cube)));
    }
}

// Elementwise residual add: out = inp1 + inp2
void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// LayerNorm forward, per-row (rows = B*T typically), C channels per row,
// with affine weight/bias (both length C).
void layernorm_forward_cpu(float* out, const float* inp,
                                          const float* weight, const float* bias,
                                          int rows, int C, float eps) {
    for (int r = 0; r < rows; r++) {
        const float* x = inp + (size_t)r * C;
        float* y = out + (size_t)r * C;
        float mean = 0.0f;
        for (int c = 0; c < C; c++) mean += x[c];
        mean /= C;
        float var = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = x[c] - mean;
            var += d * d;
        }
        var /= C;
        float rstd = 1.0f / std::sqrt(var + eps);
        for (int c = 0; c < C; c++) {
            y[c] = (x[c] - mean) * rstd * weight[c] + bias[c];
        }
    }
}

// Numerically-stable row-wise softmax (rows x C).
void softmax_forward_cpu(float* out, const float* inp, int rows, int C) {
    for (int r = 0; r < rows; r++) {
        const float* x = inp + (size_t)r * C;
        float* y = out + (size_t)r * C;
        float maxv = x[0];
        for (int c = 1; c < C; c++) if (x[c] > maxv) maxv = x[c];
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            float e = std::exp(x[c] - maxv);
            y[c] = e;
            sum += e;
        }
        for (int c = 0; c < C; c++) y[c] /= sum;
    }
}

// Causal multi-head scaled-dot-product self-attention core (GPT-2 style:
// each query position only attends to key/value positions <= itself).
//
// q, k, v, out are all [B, T, NH, HS] contiguous (i.e. [B, T, C] with
// C = NH*HS, heads interleaved in the channel dimension -- this layout
// means no explicit transpose kernel is needed between the QKV
// projection and the attention core).
void attention_forward_cpu(float* out, const float* q, const float* k,
                                          const float* v, int B, int T, int NH, int HS) {
    int C = NH * HS;
    float scale = 1.0f / std::sqrt(static_cast<float>(HS));
    float* scores = (float*)malloc(sizeof(float) * (size_t)T); // reused per (b,h,t)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            for (int t = 0; t < T; t++) {
                const float* qt = q + ((size_t)b * T + t) * C + (size_t)h * HS;

                float maxv = -1e30f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* kt2 = k + ((size_t)b * T + t2) * C + (size_t)h * HS;
                    float dot = 0.0f;
                    for (int s = 0; s < HS; s++) dot += qt[s] * kt2[s];
                    dot *= scale;
                    scores[t2] = dot;
                    if (dot > maxv) maxv = dot;
                }
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float e = std::exp(scores[t2] - maxv);
                    scores[t2] = e;
                    sum += e;
                }
                for (int t2 = 0; t2 <= t; t2++) scores[t2] /= sum;

                float* ot = out + ((size_t)b * T + t) * C + (size_t)h * HS;
                for (int s = 0; s < HS; s++) ot[s] = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* vt2 = v + ((size_t)b * T + t2) * C + (size_t)h * HS;
                    float w = scores[t2];
                    for (int s = 0; s < HS; s++) ot[s] += w * vt2[s];
                }
            }
        }
    }
    free(scores);
}

