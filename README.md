# Project 2 — CUDA transformer inference kernels

Starter kit for the second project of the Computação Paralela course.mofi

The project has two parts:

1. **Kernels (`src/project/`)**: implement the CUDA kernels for the five
   operations of a GPT-2-style transformer block (GELU, residual add,
   LayerNorm, softmax and causal multi-head attention), starting with a
   correct version and then optimizing it step by step.
2. **Pipeline (`src/assemble/`)**: optimize the end-to-end forward pass of
   one transformer block (`block_forward.cu`), which chains your kernels
   with cuBLAS matrix multiplications.

## Layout

```
include/
  project.h            launcher declarations for the kernels you implement + CUDA_CHECK + check_tensor
  reference.h          CPU reference implementations (the correctness oracle)
  linear_forward.h     cuBLAS and CPU linear layer (y = x @ W^T), provided
  rng.h                seeded random-tensor filler
src/
  project/             <-- YOUR KERNELS
    gelu_forward.cu        warm-up
    residual_forward.cu    warm-up
    layernorm_forward.cu   required
    softmax_forward.cu     required
    attention_forward.cu   required, the project's centerpiece
  assemble/            <-- YOUR PIPELINE OPTIMIZATION
    block_forward.cu       one pre-LN transformer block, GPU vs. CPU reference
    linear_forward.cu      cuBLAS linear layer (provided, do not modify)
  reference/
    reference.cpp          CPU implementations of every op (do not modify)
    dump_test_case.cpp     dumps fixed-seed test cases for reference validation
  common/
    check_tensor.cpp       tensor comparison helper used by tests and pipeline
    rng.cpp                random filler
test/                  Google Test harnesses, one per kernel (correctness + timing)
third_party/           Google Test (see Building)
CMakeLists.txt
```

Files you are expected to modify: everything in `src/project/`,
`src/assemble/block_forward.cu`, and the test files in `test/` (to add test
cases for your new kernel versions). Do **not** change the CPU reference
(`src/reference/`), `include/reference.h`, or the launcher signatures in
`include/project.h`: the tests and the pipeline depend on them.

## Part 1 — Kernels (`src/project/`)

Each `.cu` file in `src/project/` contains:

- one or more `__global__` kernels, and
- a host **launcher** function (declared in `include/project.h`) that takes
  a `version` number and dispatches to the corresponding kernel with its
  launch configuration.

The kernels are provided as **empty stubs** wired to `version 1`. Your work:

1. **Version 1: a correct, simple implementation.** Fill in the stub so
   that it matches the CPU reference in `src/reference/reference.cpp`.
   Keep it simple (e.g. one thread per element or per row); this is your
   baseline.
2. **Versions 2, 3, …: optimized implementations.** Add each new kernel as
   a new `case` in the launcher's `switch`, keeping the earlier versions so
   they can be compared. Each version should be motivated by what you
   measured on the previous one (memory coalescing, shared memory, warp
   shuffles, occupancy and block size, kernel fusion, tiling, …).

All pointers given to the launchers are **device pointers**. Data layouts:

| Op | Shapes | Notes |
|----|--------|-------|
| `gelu_forward` | `N` floats | GPT-2 tanh approximation |
| `residual_forward` | `N` floats | `out = a + b` |
| `layernorm_forward` | `[rows, C]`, weight/bias `[C]` | per-row normalization, `eps = 1e-5` |
| `softmax_forward` | `[rows, C]` | numerically stable (subtract row max) |
| `attention_forward` | q, k, v, out: `[B, T, NH, HS]` (= `[B, T, C]`, `C = NH*HS`) | causal: position `t` only attends to positions `<= t`; scale `1/sqrt(HS)`. `T` must not exceed `MAX_T` in `attention_forward.cu` |

The reference model sizes used by the tests and the pipeline are
`B = 8, T = 128, C = 256, NH = 4, HS = 64`.

### Testing a kernel

Every kernel has a Google Test harness in `test/<op>_test.cu`. It fills
random inputs with a fixed seed, runs the CPU reference and your kernel,
compares the results (`check_tensor`, tolerance 1e-3, or 1e-4 for the
residual add) and then times 100 launches with `cudaEvent`s:

```
version 1: 0.1234 ms/launch (avg over 100)
```

Only `version 1` has a test case out of the box. When you add a version,
add the corresponding test case at the bottom of the test file, e.g.:

```cpp
TEST(LayernormForward, V2) {
    constexpr auto version = 2;
    layernorm_forward_test(version);
}
```

## Part 2 — Pipeline (`src/assemble/block_forward.cu`)

`block_forward.cu` runs one pre-LN, GPT-2-style transformer block:

```
x -> LN1 -> QKV projection (cuBLAS) -> split Q/K/V -> attention
  -> output projection (cuBLAS) -> residual add (+x)
  -> LN2 -> FFN1 (cuBLAS) -> GELU -> FFN2 (cuBLAS) -> residual add
```

The same computation is run on the CPU with the reference functions and the
two outputs are compared (tolerance 2e-2, looser than the per-kernel tests
because floating-point error accumulates across the chain).

```
./bin/block_forward [version] [gpu|cpu]
```

- `version` (default 1): the kernel version passed to every launcher.
- `gpu` (default): runs one untimed warm-up pass, then times 10 GPU forward
  passes with `cudaEvent`s and prints the average. Only the forward pass is
  timed (input already on the device to output on the device), without
  allocation, host-device transfers or the CPU check. The output is then
  compared with the CPU reference (not timed).
- `cpu`: times one pass of the CPU reference chain, the baseline for your
  speedups (`version` is ignored).

```sh
./bin/block_forward 1 gpu   # every op uses its version 1 kernel
./bin/block_forward 2 gpu   # every op uses its version 2 kernel
./bin/block_forward 1 cpu   # CPU baseline
```

The given pipeline is deliberately straightforward: one kernel launch per
op, an intermediate buffer for every step, and three `cudaMemcpy2D` calls
to split the QKV projection into separate Q, K and V buffers. Your task is
to **optimize the forward pass as a whole**, beyond plugging in your best
kernels, and report the speedup of each change with the timing that
`block_forward` prints. Things worth investigating:

- choosing the best version of each kernel (they need not share a number);
- avoiding the Q/K/V split copies (e.g. let attention read the interleaved
  `[rows, 3*C]` QKV buffer directly);
- fusing consecutive operations (e.g. residual add + LayerNorm, or GELU
  into the FFN path) to save kernel launches and global-memory round trips;
- reusing intermediate buffers;
- overlapping independent work with CUDA streams, or reducing launch
  overhead with CUDA Graphs;

`linear_forward_cublas` (in `src/assemble/linear_forward.cu`) is provided
and is not part of the optimization work; use it for every matrix
multiplication. The end-to-end result must still pass the comparison with
the CPU reference.

## Building

Requirements: CMake ≥ 3.18, a CUDA 11.x toolkit (with cuBLAS) and a C++
compiler. Google Test is not included in the repository; clone it into
`third_party/` once:

```sh
git clone --depth 1 https://github.com/google/googletest.git third_party/googletest
```

Then configure and build (the GPU architecture is detected automatically):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build                         # the five kernel tests
cmake --build build --target block_forward  # the assembled pipeline
```

Executables are written to `bin/`:

```sh
./bin/gelu_forward_test
./bin/residual_forward_test
./bin/layernorm_forward_test
./bin/softmax_forward_test
./bin/attention_forward_test
./bin/attention_forward_test --gtest_filter='*V2'   # run a single version
./bin/block_forward 2 gpu
```

`CUDA_CHECK` only checks errors when compiled with `-DCHECK`. While
developing, build with error checking on, then turn it off for the timing
runs:

```sh
cmake -S . -B build -DCMAKE_CUDA_FLAGS="-DCHECK" -DCMAKE_CXX_FLAGS="-DCHECK"
```

## Profiling

Nsight Compute and Nsight Systems do **not** support this GPU (Nsight
Compute requires Volta or newer; Nsight Systems GPU tracing starts at
Pascal). Use `nvprof` or the NVIDIA Visual Profiler (`nvvp`):

```sh
nvprof --print-gpu-trace ./bin/attention_forward_test
nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency,dram_utilization \
       ./bin/attention_forward_test --gtest_filter='*V2'
nvprof ./bin/block_forward 2 gpu  # per-kernel time breakdown of the whole block
```

Combine these with the `cudaEvent` timings printed by the tests. The
deliverable is not just code that compiles: for each kernel version and
each pipeline change, report the measured effect (time and relevant
metrics) and explain why it improved, or why it did not.
