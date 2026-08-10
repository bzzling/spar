# Spar

Spar is a dependency-free C++23 deep-learning framework built from scratch, with an initial focus
on training and running Transformer language models. Its CPU implementation is a correctness-first
reference for later custom acceleration work.

## Current capabilities

- C++23 modules and an `import spar;` umbrella API
- owning and viewed CPU tensors with reverse-mode automatic differentiation
- reference tensor operations and Transformer neural-network primitives
- raw-byte BPE tokenization and deterministic language-model data iteration
- AdamW, gradient accumulation/clipping, warmup-cosine scheduling, and exact checkpoint/resume

## Build

Spar currently requires CMake 4.4+, Ninja, and a recent LLVM Clang/libc++ toolchain with C++23
standard-library module support.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

Spar's larger validation suite and examples are currently local development material and are not
shipped in the public repository. Developers with those trees available may enable
`SPAR_BUILD_TESTS` and `SPAR_BUILD_EXAMPLES`; enabling either option without its corresponding local
directory produces an intentional configuration error.

The project intentionally has no third-party ML, testing, formatting, or logging dependencies.
CUDA, mixed precision, fused kernels, distributed training, generation, and KV caching are not yet
implemented.
