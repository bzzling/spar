#include "cuda_kernels.hpp"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr unsigned int threads_per_block = 256;
constexpr unsigned int maximum_blocks = 65535;

SparCudaStatus success() {
  return {0, static_cast<int>(cudaSuccess)};
}

SparCudaStatus invalid_argument() {
  return {1, static_cast<int>(cudaErrorInvalidValue)};
}

SparCudaStatus runtime_error(cudaError_t error) {
  return {2, static_cast<int>(error)};
}

SparCudaStatus select_device(int device, int* previous, bool* changed) {
  cudaError_t error = cudaGetDevice(previous);
  if (error != cudaSuccess) {
    return runtime_error(error);
  }
  *changed = *previous != device;
  if (*changed) {
    error = cudaSetDevice(device);
    if (error != cudaSuccess) {
      return runtime_error(error);
    }
  }
  return success();
}

SparCudaStatus finish_launch(int previous, bool changed) {
  cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }
  const cudaError_t restore_error = changed ? cudaSetDevice(previous) : cudaSuccess;
  if (error != cudaSuccess) {
    return runtime_error(error);
  }
  return restore_error == cudaSuccess ? success() : runtime_error(restore_error);
}

unsigned int block_count(uint64_t count) {
  uint64_t blocks = (count + threads_per_block - 1U) / threads_per_block;
  if (blocks > maximum_blocks) {
    blocks = maximum_blocks;
  }
  return static_cast<unsigned int>(blocks);
}

template <typename T> __device__ T stable_sigmoid(T value) {
  if (value >= T{0}) {
    return T{1} / (T{1} + exp(-value));
  }
  const T exponential = exp(value);
  return exponential / (T{1} + exponential);
}

template <typename T>
__global__ void binary_kernel(T* output, const T* left, const T* right, uint64_t count,
                              int operation) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = start; index < count; index += stride) {
    switch (operation) {
    case SPAR_CUDA_ADD:
      output[index] = left[index] + right[index];
      break;
    case SPAR_CUDA_SUBTRACT:
      output[index] = left[index] - right[index];
      break;
    case SPAR_CUDA_MULTIPLY:
      output[index] = left[index] * right[index];
      break;
    case SPAR_CUDA_DIVIDE:
      output[index] = left[index] / right[index];
      break;
    }
  }
}

template <typename T>
__global__ void scalar_kernel(T* output, const T* input, uint64_t count, int operation, T scalar) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = start; index < count; index += stride) {
    switch (operation) {
    case SPAR_CUDA_ADD:
      output[index] = input[index] + scalar;
      break;
    case SPAR_CUDA_SUBTRACT:
      output[index] = input[index] - scalar;
      break;
    case SPAR_CUDA_MULTIPLY:
      output[index] = input[index] * scalar;
      break;
    case SPAR_CUDA_DIVIDE:
      output[index] = input[index] / scalar;
      break;
    }
  }
}

template <typename T>
__global__ void unary_kernel(T* output, const T* input, uint64_t count, int operation) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = start; index < count; index += stride) {
    const T value = input[index];
    switch (operation) {
    case SPAR_CUDA_NEGATE:
      output[index] = -value;
      break;
    case SPAR_CUDA_SQUARE:
      output[index] = value * value;
      break;
    case SPAR_CUDA_RECIPROCAL:
      output[index] = T{1} / value;
      break;
    case SPAR_CUDA_EXP:
      output[index] = exp(value);
      break;
    case SPAR_CUDA_LOG:
      output[index] = log(value);
      break;
    case SPAR_CUDA_SQRT:
      output[index] = sqrt(value);
      break;
    case SPAR_CUDA_SIGMOID:
      output[index] = stable_sigmoid(value);
      break;
    case SPAR_CUDA_SILU:
      output[index] = value * stable_sigmoid(value);
      break;
    }
  }
}

template <typename T>
__global__ void sum_kernel(T* output, const T* input, uint64_t count, bool take_mean) {
  __shared__ T partial[threads_per_block];
  T value = T{0};
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    value += input[index];
  }
  partial[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int width = blockDim.x / 2U; width != 0; width /= 2U) {
    if (threadIdx.x < width) {
      partial[threadIdx.x] += partial[threadIdx.x + width];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[0] = take_mean ? partial[0] / static_cast<T>(count) : partial[0];
  }
}

template <typename T> __global__ void max_kernel(T* output, const T* input, uint64_t count) {
  __shared__ T partial[threads_per_block];
  __shared__ unsigned int contains_nan[threads_per_block];
  T value = -static_cast<T>(INFINITY);
  unsigned int nan = 0;
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    const T candidate = input[index];
    if (isnan(candidate)) {
      value = candidate;
      nan = 1;
      break;
    }
    if (candidate > value) {
      value = candidate;
    }
  }
  partial[threadIdx.x] = value;
  contains_nan[threadIdx.x] = nan;
  __syncthreads();
  for (unsigned int width = blockDim.x / 2U; width != 0; width /= 2U) {
    if (threadIdx.x < width) {
      if (contains_nan[threadIdx.x + width] != 0U) {
        partial[threadIdx.x] = partial[threadIdx.x + width];
        contains_nan[threadIdx.x] = 1;
      } else if (contains_nan[threadIdx.x] == 0U &&
                 partial[threadIdx.x + width] > partial[threadIdx.x]) {
        partial[threadIdx.x] = partial[threadIdx.x + width];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[0] = partial[0];
  }
}

template <typename T>
__global__ void fill_from_scalar_kernel(T* output, const T* scalar, uint64_t count, T scale) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  const T value = scalar[0] * scale;
  for (uint64_t index = start; index < count; index += stride) {
    output[index] = value;
  }
}

template <typename T>
__global__ void add_in_place_kernel(T* destination, const T* source, uint64_t count) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = start; index < count; index += stride) {
    destination[index] += source[index];
  }
}

template <typename T>
__global__ void row_reduction_kernel(T* output, const T* input, uint64_t row_count,
                                     uint64_t reduction_count, bool take_mean) {
  __shared__ T partial[threads_per_block];
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    T value = T{0};
    const uint64_t base = row * reduction_count;
    for (uint64_t column = threadIdx.x; column < reduction_count; column += blockDim.x) {
      value += input[base + column];
    }
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2U; width != 0; width /= 2U) {
      if (threadIdx.x < width) {
        partial[threadIdx.x] += partial[threadIdx.x + width];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      output[row] = take_mean ? partial[0] / static_cast<T>(reduction_count) : partial[0];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void probability_forward_kernel(T* output, int32_t* undefined_slices, const T* input,
                                           uint64_t outer, uint64_t axis_extent, uint64_t inner,
                                           bool logarithmic) {
  __shared__ T partial_values[threads_per_block];
  __shared__ unsigned int partial_nan[threads_per_block];
  __shared__ uint64_t partial_positive_infinities[threads_per_block];
  __shared__ T slice_maximum;
  const uint64_t slice_count = outer * inner;
  const T infinity = static_cast<T>(INFINITY);
  const T nan = static_cast<T>(NAN);

  for (uint64_t slice = blockIdx.x; slice < slice_count; slice += gridDim.x) {
    const uint64_t outer_index = slice / inner;
    const uint64_t inner_index = slice % inner;
    T maximum = -infinity;
    unsigned int contains_nan = 0;
    uint64_t positive_infinities = 0;
    for (uint64_t axis_index = threadIdx.x; axis_index < axis_extent; axis_index += blockDim.x) {
      const uint64_t logical = (outer_index * axis_extent + axis_index) * inner + inner_index;
      const T value = input[logical];
      contains_nan |= isnan(value) ? 1U : 0U;
      positive_infinities += value == infinity ? 1U : 0U;
      if (value > maximum) {
        maximum = value;
      }
    }
    partial_values[threadIdx.x] = maximum;
    partial_nan[threadIdx.x] = contains_nan;
    partial_positive_infinities[threadIdx.x] = positive_infinities;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2U; width != 0; width /= 2U) {
      if (threadIdx.x < width) {
        if (partial_values[threadIdx.x + width] > partial_values[threadIdx.x]) {
          partial_values[threadIdx.x] = partial_values[threadIdx.x + width];
        }
        partial_nan[threadIdx.x] |= partial_nan[threadIdx.x + width];
        partial_positive_infinities[threadIdx.x] +=
            partial_positive_infinities[threadIdx.x + width];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      slice_maximum = partial_values[0];
    }
    __syncthreads();
    const bool undefined = partial_nan[0] != 0U || partial_values[0] == -infinity ||
                           partial_positive_infinities[0] != 0U;
    if (threadIdx.x == 0) {
      undefined_slices[slice] = undefined ? 1 : 0;
    }
    if (partial_nan[0] != 0U || partial_values[0] == -infinity) {
      for (uint64_t axis_index = threadIdx.x; axis_index < axis_extent; axis_index += blockDim.x) {
        const uint64_t logical = (outer_index * axis_extent + axis_index) * inner + inner_index;
        output[logical] = nan;
      }
      __syncthreads();
      continue;
    }
    if (partial_positive_infinities[0] != 0U) {
      const T positive_value = logarithmic ? -log(static_cast<T>(partial_positive_infinities[0]))
                                           : T{1} / static_cast<T>(partial_positive_infinities[0]);
      const T other_value = logarithmic ? -infinity : T{0};
      for (uint64_t axis_index = threadIdx.x; axis_index < axis_extent; axis_index += blockDim.x) {
        const uint64_t logical = (outer_index * axis_extent + axis_index) * inner + inner_index;
        output[logical] = input[logical] == infinity ? positive_value : other_value;
      }
      __syncthreads();
      continue;
    }

    T exponential_sum = T{0};
    for (uint64_t axis_index = threadIdx.x; axis_index < axis_extent; axis_index += blockDim.x) {
      const uint64_t logical = (outer_index * axis_extent + axis_index) * inner + inner_index;
      exponential_sum += exp(input[logical] - slice_maximum);
    }
    partial_values[threadIdx.x] = exponential_sum;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2U; width != 0; width /= 2U) {
      if (threadIdx.x < width) {
        partial_values[threadIdx.x] += partial_values[threadIdx.x + width];
      }
      __syncthreads();
    }
    const T normalization = logarithmic ? log(partial_values[0]) : partial_values[0];
    for (uint64_t axis_index = threadIdx.x; axis_index < axis_extent; axis_index += blockDim.x) {
      const uint64_t logical = (outer_index * axis_extent + axis_index) * inner + inner_index;
      output[logical] = logarithmic ? (input[logical] - slice_maximum) - normalization
                                    : exp(input[logical] - slice_maximum) / normalization;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void probability_backward_kernel(T* output, const T* gradient, const T* saved_output,
                                            const int32_t* undefined_slices, uint64_t outer,
                                            uint64_t axis_extent, uint64_t inner,
                                            bool logarithmic) {
  __shared__ T partial[threads_per_block];
  const uint64_t slice_count = outer * inner;
  const T nan = static_cast<T>(NAN);
  for (uint64_t slice = blockIdx.x; slice < slice_count; slice += gridDim.x) {
    const uint64_t outer_index = slice / inner;
    const uint64_t inner_index = slice % inner;
    if (undefined_slices[slice] != 0) {
      for (uint64_t axis_index = threadIdx.x; axis_index < axis_extent; axis_index += blockDim.x) {
        const uint64_t logical = (outer_index * axis_extent + axis_index) * inner + inner_index;
        output[logical] = nan;
      }
      __syncthreads();
      continue;
    }
    T accumulated = T{0};
    for (uint64_t axis_index = threadIdx.x; axis_index < axis_extent; axis_index += blockDim.x) {
      const uint64_t logical = (outer_index * axis_extent + axis_index) * inner + inner_index;
      accumulated += logarithmic ? gradient[logical] : gradient[logical] * saved_output[logical];
    }
    partial[threadIdx.x] = accumulated;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2U; width != 0; width /= 2U) {
      if (threadIdx.x < width) {
        partial[threadIdx.x] += partial[threadIdx.x + width];
      }
      __syncthreads();
    }
    const T reduced = partial[0];
    for (uint64_t axis_index = threadIdx.x; axis_index < axis_extent; axis_index += blockDim.x) {
      const uint64_t logical = (outer_index * axis_extent + axis_index) * inner + inner_index;
      output[logical] = logarithmic ? gradient[logical] - exp(saved_output[logical]) * reduced
                                    : saved_output[logical] * (gradient[logical] - reduced);
    }
    __syncthreads();
  }
}

bool valid_dtype(int dtype) {
  return dtype == SPAR_CUDA_FLOAT32 || dtype == SPAR_CUDA_FLOAT64;
}

} // namespace

extern "C" SparCudaStatus spar_cuda_launch_binary(void* output, const void* left, const void* right,
                                                  uint64_t count, int dtype, int operation,
                                                  int device) {
  if (!valid_dtype(dtype) || operation < SPAR_CUDA_ADD || operation > SPAR_CUDA_DIVIDE ||
      (count != 0 && (output == nullptr || left == nullptr || right == nullptr))) {
    return invalid_argument();
  }
  if (count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    binary_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(left),
        static_cast<const float*>(right), count, operation);
  } else {
    binary_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(left),
        static_cast<const double*>(right), count, operation);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_scalar(void* output, const void* input, uint64_t count,
                                                  int dtype, int operation, double scalar,
                                                  int device) {
  if (!valid_dtype(dtype) || operation < SPAR_CUDA_ADD || operation > SPAR_CUDA_DIVIDE ||
      (count != 0 && (output == nullptr || input == nullptr))) {
    return invalid_argument();
  }
  if (count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    scalar_kernel<<<blocks, threads_per_block>>>(static_cast<float*>(output),
                                                 static_cast<const float*>(input), count, operation,
                                                 static_cast<float>(scalar));
  } else {
    scalar_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(input), count, operation, scalar);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_unary(void* output, const void* input, uint64_t count,
                                                 int dtype, int operation, int device) {
  if (!valid_dtype(dtype) || operation < SPAR_CUDA_NEGATE || operation > SPAR_CUDA_SILU ||
      (count != 0 && (output == nullptr || input == nullptr))) {
    return invalid_argument();
  }
  if (count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    unary_kernel<<<blocks, threads_per_block>>>(static_cast<float*>(output),
                                                static_cast<const float*>(input), count, operation);
  } else {
    unary_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(input), count, operation);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_reduction(void* output, const void* input,
                                                     uint64_t count, int dtype, int operation,
                                                     int device) {
  if (!valid_dtype(dtype) || operation < SPAR_CUDA_SUM || operation > SPAR_CUDA_MAX ||
      output == nullptr || (count != 0 && input == nullptr) ||
      ((operation == SPAR_CUDA_MEAN || operation == SPAR_CUDA_MAX) && count == 0)) {
    return invalid_argument();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  if (dtype == SPAR_CUDA_FLOAT32) {
    if (operation == SPAR_CUDA_MAX) {
      max_kernel<<<1, threads_per_block>>>(static_cast<float*>(output),
                                           static_cast<const float*>(input), count);
    } else {
      sum_kernel<<<1, threads_per_block>>>(static_cast<float*>(output),
                                           static_cast<const float*>(input), count,
                                           operation == SPAR_CUDA_MEAN);
    }
  } else if (operation == SPAR_CUDA_MAX) {
    max_kernel<<<1, threads_per_block>>>(static_cast<double*>(output),
                                         static_cast<const double*>(input), count);
  } else {
    sum_kernel<<<1, threads_per_block>>>(static_cast<double*>(output),
                                         static_cast<const double*>(input), count,
                                         operation == SPAR_CUDA_MEAN);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_row_reduction(void* output, const void* input,
                                                         uint64_t row_count,
                                                         uint64_t reduction_count, int dtype,
                                                         int operation, int device) {
  if (!valid_dtype(dtype) || (operation != SPAR_CUDA_SUM && operation != SPAR_CUDA_MEAN) ||
      (row_count != 0 && output == nullptr) ||
      (row_count != 0 && reduction_count != 0 && input == nullptr) ||
      (operation == SPAR_CUDA_MEAN && reduction_count == 0)) {
    return invalid_argument();
  }
  if (row_count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(row_count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    row_reduction_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(input), row_count, reduction_count,
        operation == SPAR_CUDA_MEAN);
  } else {
    row_reduction_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(input), row_count, reduction_count,
        operation == SPAR_CUDA_MEAN);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus
spar_cuda_launch_probability_forward(void* output, int32_t* undefined_slices, const void* input,
                                     uint64_t outer, uint64_t axis_extent, uint64_t inner,
                                     int dtype, int logarithmic, int device) {
  if (inner == 0 || outer > UINT64_MAX / inner) {
    return invalid_argument();
  }
  const uint64_t slice_count = outer * inner;
  if (!valid_dtype(dtype) || axis_extent == 0 || (logarithmic != 0 && logarithmic != 1) ||
      slice_count == 0 || output == nullptr || undefined_slices == nullptr || input == nullptr) {
    return invalid_argument();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(slice_count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    probability_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), undefined_slices, static_cast<const float*>(input), outer,
        axis_extent, inner, logarithmic != 0);
  } else {
    probability_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), undefined_slices, static_cast<const double*>(input), outer,
        axis_extent, inner, logarithmic != 0);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_probability_backward(
    void* output, const void* gradient, const void* saved_output, const int32_t* undefined_slices,
    uint64_t outer, uint64_t axis_extent, uint64_t inner, int dtype, int logarithmic, int device) {
  if (inner == 0 || outer > UINT64_MAX / inner) {
    return invalid_argument();
  }
  const uint64_t slice_count = outer * inner;
  if (!valid_dtype(dtype) || axis_extent == 0 || (logarithmic != 0 && logarithmic != 1) ||
      slice_count == 0 || output == nullptr || gradient == nullptr || saved_output == nullptr ||
      undefined_slices == nullptr) {
    return invalid_argument();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(slice_count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    probability_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(gradient),
        static_cast<const float*>(saved_output), undefined_slices, outer, axis_extent, inner,
        logarithmic != 0);
  } else {
    probability_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(gradient),
        static_cast<const double*>(saved_output), undefined_slices, outer, axis_extent, inner,
        logarithmic != 0);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_fill_from_scalar(void* output, const void* scalar,
                                                            uint64_t count, int dtype, double scale,
                                                            int device) {
  if (!valid_dtype(dtype) || scalar == nullptr || (count != 0 && output == nullptr)) {
    return invalid_argument();
  }
  if (count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    fill_from_scalar_kernel<<<blocks, threads_per_block>>>(static_cast<float*>(output),
                                                           static_cast<const float*>(scalar), count,
                                                           static_cast<float>(scale));
  } else {
    fill_from_scalar_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(scalar), count, scale);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_add_in_place(void* destination, const void* source,
                                                        uint64_t count, int dtype, int device) {
  if (!valid_dtype(dtype) || (count != 0 && (destination == nullptr || source == nullptr))) {
    return invalid_argument();
  }
  if (count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    add_in_place_kernel<<<blocks, threads_per_block>>>(static_cast<float*>(destination),
                                                       static_cast<const float*>(source), count);
  } else {
    add_in_place_kernel<<<blocks, threads_per_block>>>(static_cast<double*>(destination),
                                                       static_cast<const double*>(source), count);
  }
  return finish_launch(previous, changed);
}

extern "C" const char* spar_cuda_error_string(int cuda_error) {
  return cudaGetErrorString(static_cast<cudaError_t>(cuda_error));
}
