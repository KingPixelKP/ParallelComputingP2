// Host-only (no CUDA) sanity check for common/reference.h.
//
// Dumps small, fixed-seed random test cases and the reference
// implementation's outputs for each op to validate/data/*.txt.

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "rng.h"
#include "reference.h"

static void write_vec(const std::string& path, const float* a, int n) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << path << ": cannot open file for writing" << std::endl;
        std::exit(1);
    }
    f << std::setprecision(8);
    for (int i = 0; i < n; i++) f << a[i] << '\n';
}

int main(int argc, char* argv[]) {

    const std::string path = (argc > 1) ? argv[1] : "../validate";

    std::cout << "Validation folder: " << path << std::endl;
    std::srand(42);


    // ---- GELU / residual: flat vector of N=24 ----
    {
        int N = 24;
        float *x = (float*) malloc(N * sizeof(float));
        float *y = (float*) malloc(N * sizeof(float));
        float *out = (float*) malloc(N * sizeof(float));
        fill_random(x, N, -2.0f, 2.0f);
        fill_random(y, N, -2.0f, 2.0f);

        write_vec(path + "/data/gelu_in.txt", x, N);
        gelu_forward_cpu(out, x, N);
        write_vec(path + "/data/gelu_out.txt", out, N);

        residual_forward_cpu(out, x, y, N);
        write_vec(path + "/data/residual_a.txt", x, N);
        write_vec(path + "/data/residual_b.txt", y, N);
        write_vec(path + "/data/residual_out.txt", out, N);

        free(x); free(y); free(out);
    }

    // ---- LayerNorm / Softmax: rows=4, C=6 ----
    {
        int rows = 4, C = 6, N = rows * C;
        float *x = (float*) malloc(N * sizeof(float));
        float *w = (float*) malloc(C * sizeof(float));
        float *b = (float*) malloc(C * sizeof(float));
        float *out = (float*) malloc(N * sizeof(float));
        fill_random(x, N, -2.0f, 2.0f);
        fill_random(w, C, 0.5f, 1.5f);
        fill_random(b, C, -0.5f, 0.5f);

        write_vec(path + "/data/ln_in.txt", x, N);
        write_vec(path + "/data/ln_weight.txt", w, C);
        write_vec(path + "/data/ln_bias.txt", b, C);
        layernorm_forward_cpu(out, x, w, b, rows, C, 1e-5f);
        write_vec(path + "/data/ln_out.txt", out, N);

        softmax_forward_cpu(out, x, rows, C);
        write_vec(path + "/data/softmax_in.txt", x, N);
        write_vec(path + "/data/softmax_out.txt", out, N);

        free(x); free(w); free(b); free(out);
    }

    // ---- Attention: B=2, T=5, NH=2, HS=3 (C=6), causal ----
    {
        int B = 2, T = 5, NH = 2, HS = 3, C = NH * HS;
        int N = B * T * C;
        float *q = (float*) malloc(N * sizeof(float));
        float *k = (float*) malloc(N * sizeof(float));
        float *v = (float*) malloc(N * sizeof(float));
        float *out = (float*)  malloc(N * sizeof(float));
        fill_random(q, N, -1.0f, 1.0f);
        fill_random(k, N, -1.0f, 1.0f);
        fill_random(v, N, -1.0f, 1.0f);

        write_vec(path + "/data/attn_q.txt", q, N);
        write_vec(path + "/data/attn_k.txt", k, N);
        write_vec(path + "/data/attn_v.txt", v, N);
        attention_forward_cpu(out, q, k, v, B, T, NH, HS);
        write_vec(path + "/data/attn_out.txt", out, N);

        free(q); free(k); free(v); free(out);
    }

    std::cout << "Dumped reference test cases to " << path << "/data/" << std::endl;
    return 0;
}
