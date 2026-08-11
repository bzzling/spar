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
