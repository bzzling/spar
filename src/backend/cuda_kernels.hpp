#ifndef SPAR_BACKEND_CUDA_KERNELS_HPP
#define SPAR_BACKEND_CUDA_KERNELS_HPP

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum SparCudaDType {
  SPAR_CUDA_FLOAT32 = 0,
  SPAR_CUDA_FLOAT64 = 1,
};

enum SparCudaBinaryOp {
  SPAR_CUDA_ADD = 0,
  SPAR_CUDA_SUBTRACT = 1,
  SPAR_CUDA_MULTIPLY = 2,
  SPAR_CUDA_DIVIDE = 3,
};

enum SparCudaUnaryOp {
  SPAR_CUDA_NEGATE = 0,
  SPAR_CUDA_SQUARE = 1,
  SPAR_CUDA_RECIPROCAL = 2,
  SPAR_CUDA_EXP = 3,
  SPAR_CUDA_LOG = 4,
  SPAR_CUDA_SQRT = 5,
  SPAR_CUDA_SIGMOID = 6,
  SPAR_CUDA_SILU = 7,
};

enum SparCudaReductionOp {
  SPAR_CUDA_SUM = 0,
  SPAR_CUDA_MEAN = 1,
  SPAR_CUDA_MAX = 2,
};

struct SparCudaStatus {
  int code;
  int cuda_error;
};

struct SparCudaStatus spar_cuda_launch_binary(void* output, const void* left, const void* right,
                                              uint64_t count, int dtype, int operation, int device);
struct SparCudaStatus spar_cuda_launch_scalar(void* output, const void* input, uint64_t count,
                                              int dtype, int operation, double scalar, int device);
struct SparCudaStatus spar_cuda_launch_unary(void* output, const void* input, uint64_t count,
                                             int dtype, int operation, int device);
struct SparCudaStatus spar_cuda_launch_reduction(void* output, const void* input, uint64_t count,
                                                 int dtype, int operation, int device);
struct SparCudaStatus spar_cuda_launch_fill_from_scalar(void* output, const void* scalar,
                                                        uint64_t count, int dtype, double scale,
                                                        int device);
struct SparCudaStatus spar_cuda_launch_add_in_place(void* destination, const void* source,
                                                    uint64_t count, int dtype, int device);
const char* spar_cuda_error_string(int cuda_error);

#ifdef __cplusplus
}
#endif

#endif
