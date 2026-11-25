#ifndef COMPAR2627_TRANSFORMER_REFERENCE_H
#define COMPAR2627_TRANSFORMER_REFERENCE_H

// GELU, GPT-2's tanh approximation.
void gelu_forward_cpu(float* out, const float* inp, int N);

// Elementwise residual add: out = inp1 + inp2
void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N);

// LayerNorm forward, per-row (rows = B*T typically), C channels per row,
// with affine weight/bias (both length C).
void layernorm_forward_cpu(float* out, const float* inp,
                                          const float* weight, const float* bias,
                                          int rows, int C, float eps) ;

// Numerically-stable row-wise softmax (rows x C).
void softmax_forward_cpu(float* out, const float* inp, int rows, int C);

// Causal multi-head scaled-dot-product self-attention core (GPT-2 style:
// each query position only attends to key/value positions <= itself).
//
// q, k, v, out are all [B, T, NH, HS] contiguous (i.e. [B, T, C] with
// C = NH*HS, heads interleaved in the channel dimension -- this layout
// means no explicit transpose kernel is needed between the QKV
// projection and the attention core).
void attention_forward_cpu(float* out, const float* q, const float* k,
                                          const float* v, int B, int T, int NH, int HS);
#endif //COMPAR2627_TRANSFORMER_PROJECT2_REFERENCE_H
