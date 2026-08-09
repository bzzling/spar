# Spar

Spar is a dependency-free C++23 deep-learning framework built from scratch, with an initial focus
on training and running Transformer language models. Its CPU implementation is a correctness-first
reference for later custom acceleration work.

Leda is the science/biomedical model family being developed with Spar. The current Leda v0 code is
a small reference definition and training harness, not a released pretrained model.

## Current capabilities

- C++23 modules and an `import spar;` umbrella API
- owning and viewed CPU tensors with reverse-mode automatic differentiation
- reference tensor operations and Transformer neural-network primitives
- raw-byte BPE tokenization and deterministic language-model data iteration
- AdamW, gradient accumulation/clipping, warmup-cosine scheduling, and exact checkpoint/resume
- Leda v0 tiny/small configurations, accounting, training/evaluation helpers, and CPU profiling

See [the Leda v0 specification](docs/leda_v0.md) for its exact architecture, reproducible reference
recipe, parameter counts, memory estimate, and current limitations.

## Build

Spar currently requires CMake 4.4+, Ninja, and a recent LLVM Clang/libc++ toolchain with C++23
standard-library module support.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DSPAR_BUILD_TESTS=ON \
  -DSPAR_BUILD_EXAMPLES=ON
cmake --build build
ctest --test-dir build --output-on-failure
./build/examples/spar_tensor_example
./build/examples/spar_leda_train
./build/examples/spar_leda_profile
```

The project intentionally has no third-party ML, testing, formatting, or logging dependencies.
CUDA, mixed precision, fused kernels, distributed training, generation, and KV caching are not yet
implemented.
