#ifndef SPAR_BACKEND_CUDA_KERNELS_HPP
#define SPAR_BACKEND_CUDA_KERNELS_HPP

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum SparCudaDType {
  SPAR_CUDA_FLOAT32 = 0,
  SPAR_CUDA_FLOAT64 = 1,
  SPAR_CUDA_INT32 = 2,
  SPAR_CUDA_INT64 = 3,
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
struct SparCudaStatus spar_cuda_launch_row_reduction(void* output, const void* input,
                                                     uint64_t row_count, uint64_t reduction_count,
                                                     int dtype, int operation, int device);
struct SparCudaStatus spar_cuda_launch_probability_forward(void* output, int32_t* undefined_slices,
                                                           const void* input, uint64_t outer,
                                                           uint64_t axis_extent, uint64_t inner,
                                                           int dtype, int logarithmic, int device);
struct SparCudaStatus spar_cuda_launch_probability_backward(
    void* output, const void* gradient, const void* saved_output, const int32_t* undefined_slices,
    uint64_t outer, uint64_t axis_extent, uint64_t inner, int dtype, int logarithmic, int device);
struct SparCudaStatus
spar_cuda_launch_embedding_forward(void* output, int32_t* invalid_index, const void* weight,
                                   const void* indices, uint64_t index_count,
                                   uint64_t vocabulary_size, uint64_t embedding_dimension,
                                   int weight_dtype, int index_dtype, int device);
struct SparCudaStatus spar_cuda_launch_embedding_backward(void* output, const void* gradient,
                                                          const void* indices, uint64_t index_count,
                                                          uint64_t embedding_dimension,
                                                          int weight_dtype, int index_dtype,
                                                          int device);
struct SparCudaStatus spar_cuda_launch_rope(void* output, const void* input, uint64_t pair_count,
                                            uint64_t sequence_length, uint64_t feature_count,
                                            uint64_t start_position, double theta, int dtype,
                                            int inverse, int device);
struct SparCudaStatus spar_cuda_launch_causal_mask(void* output, uint64_t sequence_length,
                                                   int dtype, int device);
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
