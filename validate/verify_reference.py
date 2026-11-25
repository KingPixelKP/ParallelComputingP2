#!/usr/bin/env python3
"""
Independently recomputes each op with numpy from the dumped inputs in
validate/data/, and compares against common/reference.h's dumped
outputs (produced by validate/dump_test_case.c). This cross-checks the
C reference oracle against a from-scratch numpy implementation written
without reusing the C code's control flow, so a bug shared between "the
oracle" and "the test" is unlikely.

Run via `make validate` (builds + runs dump_test_case.c first), or
directly with `python3 validate/verify_reference.py` once
validate/data/*.txt already exists.
"""
import os
import sys

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")


def load(name):
    return np.loadtxt(os.path.join(DATA, name), dtype=np.float64)


def check(name, actual, expected, tol=1e-4):
    diff = np.abs(actual - expected)
    maxdiff = diff.max()
    ok = maxdiff <= tol
    print(f"[{name}] {'PASS' if ok else 'FAIL'}  (max abs diff = {maxdiff:.6g}, tol = {tol:.6g})")
    return ok


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax_rows(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def layernorm_rows(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    return (x - mean) * rstd * w + b


def causal_attention(q, k, v, B, T, NH, HS):
    C = NH * HS
    q = q.reshape(B, T, NH, HS)
    k = k.reshape(B, T, NH, HS)
    v = v.reshape(B, T, NH, HS)
    out = np.zeros_like(q)
    scale = 1.0 / np.sqrt(HS)
    for b in range(B):
        for h in range(NH):
            qh, kh, vh = q[b, :, h, :], k[b, :, h, :], v[b, :, h, :]
            scores = (qh @ kh.T) * scale  # [T, T]
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask, -1e30, scores)
            weights = softmax_rows(scores)
            out[b, :, h, :] = weights @ vh
    return out.reshape(B, T, C)


def main():
    ok = True

    # GELU
    x = load("gelu_in.txt")
    ok &= check("gelu", gelu(x), load("gelu_out.txt"))

    # Residual
    a, b = load("residual_a.txt"), load("residual_b.txt")
    ok &= check("residual", a + b, load("residual_out.txt"))

    # LayerNorm (rows=4, C=6)
    x = load("ln_in.txt").reshape(4, 6)
    w, b_ = load("ln_weight.txt"), load("ln_bias.txt")
    expected = load("ln_out.txt").reshape(4, 6)
    ok &= check("layernorm", layernorm_rows(x, w, b_), expected)

    # Softmax (rows=4, C=6)
    x = load("softmax_in.txt").reshape(4, 6)
    expected = load("softmax_out.txt").reshape(4, 6)
    ok &= check("softmax", softmax_rows(x), expected)

    # Attention (B=2, T=5, NH=2, HS=3)
    B, T, NH, HS = 2, 5, 2, 3
    q, k, v = load("attn_q.txt"), load("attn_k.txt"), load("attn_v.txt")
    expected = load("attn_out.txt").reshape(B, T, NH * HS)
    ok &= check("attention", causal_attention(q, k, v, B, T, NH, HS), expected)

    print()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
