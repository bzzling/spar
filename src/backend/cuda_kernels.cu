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

SparCudaStatus finish_launch_and_free(void* allocation, int previous, bool changed) {
  cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }
  const cudaError_t free_error = allocation == nullptr ? cudaSuccess : cudaFree(allocation);
  const cudaError_t restore_error = changed ? cudaSetDevice(previous) : cudaSuccess;
  if (error != cudaSuccess) {
    return runtime_error(error);
  }
  if (free_error != cudaSuccess) {
    return runtime_error(free_error);
  }
  return restore_error == cudaSuccess ? success() : runtime_error(restore_error);
}

SparCudaStatus restore_after_setup_failure(cudaError_t error, void* allocation, int previous,
                                           bool changed) {
  if (allocation != nullptr) {
    static_cast<void>(cudaFree(allocation));
  }
  const cudaError_t restore_error = changed ? cudaSetDevice(previous) : cudaSuccess;
  return runtime_error(error != cudaSuccess ? error : restore_error);
}

SparCudaStatus finish_launch_copy_and_free(void* host_output, const void* device_output,
                                           size_t bytes, void* allocation, int previous,
                                           bool changed) {
  cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) {
    error = cudaDeviceSynchronize();
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(host_output, device_output, bytes, cudaMemcpyDeviceToHost);
  }
  const cudaError_t free_error = allocation == nullptr ? cudaSuccess : cudaFree(allocation);
  const cudaError_t restore_error = changed ? cudaSetDevice(previous) : cudaSuccess;
  if (error != cudaSuccess) {
    return runtime_error(error);
  }
  if (free_error != cudaSuccess) {
    return runtime_error(free_error);
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

template <typename Payload>
__global__ void strided_copy_kernel(Payload* destination, const Payload* source,
                                    const uint64_t* extents, const uint64_t* strides, uint64_t rank,
                                    uint64_t storage_offset, uint64_t count) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t logical = start; logical < count; logical += step) {
    uint64_t remainder = logical;
    uint64_t source_index = storage_offset;
    for (uint64_t index = rank; index != 0; --index) {
      const uint64_t axis = index - 1U;
      const uint64_t coordinate = remainder % extents[axis];
      remainder /= extents[axis];
      source_index += coordinate * strides[axis];
    }
    destination[logical] = source[source_index];
  }
}

template <typename T>
__global__ void
broadcast_reduce_kernel(T* destination, const T* gradient, const uint64_t* gradient_extents,
                        const uint64_t* original_extents, const uint64_t* original_strides,
                        uint64_t gradient_rank, uint64_t original_rank, uint64_t count) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  const uint64_t leading = gradient_rank - original_rank;
  for (uint64_t logical = start; logical < count; logical += step) {
    uint64_t remainder = logical;
    uint64_t destination_index = 0;
    for (uint64_t index = gradient_rank; index != 0; --index) {
      const uint64_t axis = index - 1U;
      const uint64_t coordinate = remainder % gradient_extents[axis];
      remainder /= gradient_extents[axis];
      if (axis >= leading) {
        const uint64_t original_axis = axis - leading;
        if (original_extents[original_axis] != 1U) {
          destination_index += coordinate * original_strides[original_axis];
        }
      }
    }
    atomicAdd(destination + destination_index, gradient[logical]);
  }
}

template <typename Index>
__global__ void validate_targets_kernel(int32_t* invalid_target, const Index* targets,
                                        uint64_t count, uint64_t classes) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = start; index < count; index += step) {
    const Index target = targets[index];
    if (target < 0 || static_cast<uint64_t>(target) >= classes) {
      atomicExch(invalid_target, 1);
    }
  }
}

template <typename T, typename Index>
__global__ void nll_forward_kernel(T* output, const T* log_probabilities, const Index* targets,
                                   uint64_t output_count, uint64_t classes,
                                   uint64_t sequence_length, bool language_model) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = start; index < output_count; index += step) {
    uint64_t target_index = index;
    uint64_t source_row = index;
    if (language_model) {
      const uint64_t prediction_length = sequence_length - 1U;
      const uint64_t batch = index / prediction_length;
      const uint64_t time = index % prediction_length;
      target_index = batch * sequence_length + time + 1U;
      source_row = batch * sequence_length + time;
    }
    output[index] =
        -log_probabilities[source_row * classes + static_cast<uint64_t>(targets[target_index])];
  }
}

template <typename T, typename Index>
__global__ void nll_backward_kernel(T* output, const T* gradient, const Index* targets,
                                    uint64_t output_count, uint64_t classes,
                                    uint64_t sequence_length, bool language_model) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = start; index < output_count; index += step) {
    uint64_t target_index = index;
    uint64_t source_row = index;
    if (language_model) {
      const uint64_t prediction_length = sequence_length - 1U;
      const uint64_t batch = index / prediction_length;
      const uint64_t time = index % prediction_length;
      target_index = batch * sequence_length + time + 1U;
      source_row = batch * sequence_length + time;
    }
    output[source_row * classes + static_cast<uint64_t>(targets[target_index])] = -gradient[index];
  }
}

__device__ SparCudaNormSummary combine_norm_summaries(SparCudaNormSummary left,
                                                      SparCudaNormSummary right) {
  left.has_nan |= right.has_nan;
  left.has_infinity |= right.has_infinity;
  if (right.scale == 0.0) {
    return left;
  }
  if (left.scale == 0.0) {
    left.scale = right.scale;
    left.scaled_sum_squares = right.scaled_sum_squares;
    return left;
  }
  if (left.scale < right.scale) {
    const double ratio = left.scale / right.scale;
    left.scaled_sum_squares = right.scaled_sum_squares + left.scaled_sum_squares * ratio * ratio;
    left.scale = right.scale;
  } else {
    const double ratio = right.scale / left.scale;
    left.scaled_sum_squares += right.scaled_sum_squares * ratio * ratio;
  }
  return left;
}

template <typename T>
__global__ void gradient_norm_kernel(SparCudaNormSummary* output, const T* gradient,
                                     uint64_t count) {
  __shared__ SparCudaNormSummary partial[threads_per_block];
  SparCudaNormSummary summary{0.0, 1.0, 0, 0};
  for (uint64_t index = threadIdx.x; index < count; index += blockDim.x) {
    const double magnitude = fabs(static_cast<double>(gradient[index]));
    if (isnan(magnitude)) {
      summary.has_nan = 1;
    } else if (isinf(magnitude)) {
      summary.has_infinity = 1;
    } else if (magnitude != 0.0) {
      if (summary.scale < magnitude) {
        const double ratio = summary.scale / magnitude;
        summary.scaled_sum_squares = 1.0 + summary.scaled_sum_squares * ratio * ratio;
        summary.scale = magnitude;
      } else {
        const double ratio = magnitude / summary.scale;
        summary.scaled_sum_squares += ratio * ratio;
      }
    }
  }
  partial[threadIdx.x] = summary;
  __syncthreads();
  for (unsigned int width = blockDim.x / 2U; width != 0; width /= 2U) {
    if (threadIdx.x < width) {
      partial[threadIdx.x] =
          combine_norm_summaries(partial[threadIdx.x], partial[threadIdx.x + width]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[0] = partial[0];
  }
}

template <typename T>
__global__ void scale_in_place_kernel(T* values, uint64_t count, double factor) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = start; index < count; index += step) {
    values[index] = static_cast<T>(static_cast<double>(values[index]) * factor);
  }
}

template <typename T>
__global__ void adamw_kernel(T* parameter, const T* gradient, T* first_moment, T* second_moment,
                             uint64_t count, double learning_rate, double beta1, double beta2,
                             double epsilon, double weight_decay, double first_correction,
                             double second_correction) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  const double decay_factor = 1.0 - learning_rate * weight_decay;
  for (uint64_t index = start; index < count; index += step) {
    const double gradient_value = static_cast<double>(gradient[index]);
    first_moment[index] = static_cast<T>(beta1 * static_cast<double>(first_moment[index]) +
                                         (1.0 - beta1) * gradient_value);
    second_moment[index] = static_cast<T>(beta2 * static_cast<double>(second_moment[index]) +
                                          (1.0 - beta2) * gradient_value * gradient_value);
    const double corrected_first = static_cast<double>(first_moment[index]) / first_correction;
    const double corrected_second = static_cast<double>(second_moment[index]) / second_correction;
    const double decayed = static_cast<double>(parameter[index]) * decay_factor;
    parameter[index] = static_cast<T>(decayed - learning_rate * corrected_first /
                                                    (sqrt(corrected_second) + epsilon));
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

template <typename Index>
__global__ void validate_embedding_indices_kernel(int32_t* invalid_index, const Index* indices,
                                                  uint64_t index_count, uint64_t vocabulary_size) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t position = start; position < index_count; position += stride) {
    const Index token = indices[position];
    if (token < 0 || static_cast<uint64_t>(token) >= vocabulary_size) {
      atomicExch(invalid_index, 1);
    }
  }
}

template <typename T, typename Index>
__global__ void embedding_forward_kernel(T* output, int32_t* invalid_index, const T* weight,
                                         const Index* indices, uint64_t output_count,
                                         uint64_t vocabulary_size, uint64_t embedding_dimension) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t output_index = start; output_index < output_count; output_index += stride) {
    const uint64_t position = output_index / embedding_dimension;
    const uint64_t feature = output_index % embedding_dimension;
    const Index token = indices[position];
    if (token < 0 || static_cast<uint64_t>(token) >= vocabulary_size) {
      atomicExch(invalid_index, 1);
      continue;
    }
    output[output_index] = weight[static_cast<uint64_t>(token) * embedding_dimension + feature];
  }
}

template <typename T, typename Index>
__global__ void embedding_backward_kernel(T* output, const T* gradient, const Index* indices,
                                          uint64_t gradient_count, uint64_t embedding_dimension) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t gradient_index = start; gradient_index < gradient_count; gradient_index += stride) {
    const uint64_t position = gradient_index / embedding_dimension;
    const uint64_t feature = gradient_index % embedding_dimension;
    const uint64_t token = static_cast<uint64_t>(indices[position]);
    atomicAdd(output + token * embedding_dimension + feature, gradient[gradient_index]);
  }
}

template <typename T>
__global__ void rope_kernel(T* output, const T* input, uint64_t pair_count,
                            uint64_t sequence_length, uint64_t feature_count,
                            uint64_t start_position, double theta, bool inverse) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  const uint64_t pairs_per_token = feature_count / 2U;
  for (uint64_t logical_pair = start; logical_pair < pair_count; logical_pair += stride) {
    const uint64_t pair = logical_pair % pairs_per_token;
    const uint64_t token = (logical_pair / pairs_per_token) % sequence_length;
    const uint64_t index = (logical_pair / pairs_per_token) * feature_count + 2U * pair;
    const double position = static_cast<double>(start_position + token);
    const double exponent = -2.0 * static_cast<double>(pair) / static_cast<double>(feature_count);
    const double angle = position * pow(theta, exponent);
    const T cosine = static_cast<T>(cos(angle));
    T sine = static_cast<T>(sin(angle));
    if (inverse) {
      sine = -sine;
    }
    const T first = input[index];
    const T second = input[index + 1U];
    output[index] = first * cosine - second * sine;
    output[index + 1U] = first * sine + second * cosine;
  }
}

template <typename T>
__global__ void causal_mask_kernel(T* output, uint64_t count, uint64_t sequence_length) {
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  const T negative_infinity = -static_cast<T>(INFINITY);
  for (uint64_t index = start; index < count; index += stride) {
    const uint64_t row = index / sequence_length;
    const uint64_t column = index % sequence_length;
    output[index] = column <= row ? T{0} : negative_infinity;
  }
}

bool valid_dtype(int dtype) {
  return dtype == SPAR_CUDA_FLOAT32 || dtype == SPAR_CUDA_FLOAT64;
}

bool valid_index_dtype(int dtype) {
  return dtype == SPAR_CUDA_INT32 || dtype == SPAR_CUDA_INT64;
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

extern "C" SparCudaStatus spar_cuda_launch_validate_embedding_indices(int32_t* invalid_index,
                                                                      const void* indices,
                                                                      uint64_t index_count,
                                                                      uint64_t vocabulary_size,
                                                                      int index_dtype, int device) {
  if (!valid_index_dtype(index_dtype) || index_count == 0 || invalid_index == nullptr ||
      indices == nullptr) {
    return invalid_argument();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(index_count);
  if (index_dtype == SPAR_CUDA_INT32) {
    validate_embedding_indices_kernel<<<blocks, threads_per_block>>>(
        invalid_index, static_cast<const int32_t*>(indices), index_count, vocabulary_size);
  } else {
    validate_embedding_indices_kernel<<<blocks, threads_per_block>>>(
        invalid_index, static_cast<const int64_t*>(indices), index_count, vocabulary_size);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus
spar_cuda_launch_embedding_forward(void* output, int32_t* invalid_index, const void* weight,
                                   const void* indices, uint64_t index_count,
                                   uint64_t vocabulary_size, uint64_t embedding_dimension,
                                   int weight_dtype, int index_dtype, int device) {
  if (!valid_dtype(weight_dtype) || !valid_index_dtype(index_dtype) || vocabulary_size == 0 ||
      embedding_dimension == 0 || invalid_index == nullptr ||
      index_count > UINT64_MAX / embedding_dimension) {
    return invalid_argument();
  }
  const uint64_t output_count = index_count * embedding_dimension;
  if (output_count == 0) {
    return success();
  }
  if (output == nullptr || weight == nullptr || indices == nullptr) {
    return invalid_argument();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(output_count);
  if (weight_dtype == SPAR_CUDA_FLOAT32 && index_dtype == SPAR_CUDA_INT32) {
    embedding_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), invalid_index, static_cast<const float*>(weight),
        static_cast<const int32_t*>(indices), output_count, vocabulary_size, embedding_dimension);
  } else if (weight_dtype == SPAR_CUDA_FLOAT32) {
    embedding_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), invalid_index, static_cast<const float*>(weight),
        static_cast<const int64_t*>(indices), output_count, vocabulary_size, embedding_dimension);
  } else if (index_dtype == SPAR_CUDA_INT32) {
    embedding_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), invalid_index, static_cast<const double*>(weight),
        static_cast<const int32_t*>(indices), output_count, vocabulary_size, embedding_dimension);
  } else {
    embedding_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), invalid_index, static_cast<const double*>(weight),
        static_cast<const int64_t*>(indices), output_count, vocabulary_size, embedding_dimension);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus
spar_cuda_launch_embedding_backward(void* output, const void* gradient, const void* indices,
                                    uint64_t index_count, uint64_t embedding_dimension,
                                    int weight_dtype, int index_dtype, int device) {
  if (!valid_dtype(weight_dtype) || !valid_index_dtype(index_dtype) || embedding_dimension == 0 ||
      index_count > UINT64_MAX / embedding_dimension) {
    return invalid_argument();
  }
  const uint64_t gradient_count = index_count * embedding_dimension;
  if (gradient_count == 0) {
    return success();
  }
  if (output == nullptr || gradient == nullptr || indices == nullptr) {
    return invalid_argument();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(gradient_count);
  if (weight_dtype == SPAR_CUDA_FLOAT32 && index_dtype == SPAR_CUDA_INT32) {
    embedding_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(gradient),
        static_cast<const int32_t*>(indices), gradient_count, embedding_dimension);
  } else if (weight_dtype == SPAR_CUDA_FLOAT32) {
    embedding_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(gradient),
        static_cast<const int64_t*>(indices), gradient_count, embedding_dimension);
  } else if (index_dtype == SPAR_CUDA_INT32) {
    embedding_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(gradient),
        static_cast<const int32_t*>(indices), gradient_count, embedding_dimension);
  } else {
    embedding_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(gradient),
        static_cast<const int64_t*>(indices), gradient_count, embedding_dimension);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_rope(void* output, const void* input,
                                                uint64_t pair_count, uint64_t sequence_length,
                                                uint64_t feature_count, uint64_t start_position,
                                                double theta, int dtype, int inverse, int device) {
  if (!valid_dtype(dtype) || feature_count == 0 || feature_count % 2U != 0 ||
      (inverse != 0 && inverse != 1) || !(theta > 0.0) || !isfinite(theta)) {
    return invalid_argument();
  }
  if (pair_count == 0) {
    return success();
  }
  if (sequence_length == 0 || output == nullptr || input == nullptr) {
    return invalid_argument();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(pair_count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    rope_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(input), pair_count, sequence_length,
        feature_count, start_position, theta, inverse != 0);
  } else {
    rope_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(input), pair_count,
        sequence_length, feature_count, start_position, theta, inverse != 0);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_causal_mask(void* output, uint64_t sequence_length,
                                                       int dtype, int device) {
  if (!valid_dtype(dtype) || sequence_length == 0 ||
      sequence_length > UINT64_MAX / sequence_length || output == nullptr) {
    return invalid_argument();
  }
  const uint64_t count = sequence_length * sequence_length;
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(count);
  if (dtype == SPAR_CUDA_FLOAT32) {
    causal_mask_kernel<<<blocks, threads_per_block>>>(static_cast<float*>(output), count,
                                                      sequence_length);
  } else {
    causal_mask_kernel<<<blocks, threads_per_block>>>(static_cast<double*>(output), count,
                                                      sequence_length);
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

extern "C" SparCudaStatus spar_cuda_launch_strided_copy(void* destination, const void* source,
                                                        uint64_t rank, const uint64_t* extents,
                                                        const uint64_t* strides,
                                                        uint64_t storage_offset, uint64_t count,
                                                        uint64_t element_size, int device) {
  if ((element_size != 4U && element_size != 8U) ||
      (count != 0 && (destination == nullptr || source == nullptr)) ||
      (rank != 0 && (extents == nullptr || strides == nullptr)) || rank > UINT64_MAX / 2U ||
      rank * 2U > SIZE_MAX / sizeof(uint64_t)) {
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
  uint64_t* metadata = nullptr;
  if (rank != 0) {
    cudaError_t error = cudaMalloc(&metadata, static_cast<size_t>(rank * 2U) * sizeof(uint64_t));
    if (error == cudaSuccess) {
      error = cudaMemcpy(metadata, extents, static_cast<size_t>(rank) * sizeof(uint64_t),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(metadata + rank, strides, static_cast<size_t>(rank) * sizeof(uint64_t),
                         cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) {
      return restore_after_setup_failure(error, metadata, previous, changed);
    }
  }
  const unsigned int blocks = block_count(count);
  if (element_size == 4U) {
    strided_copy_kernel<<<blocks, threads_per_block>>>(
        static_cast<uint32_t*>(destination), static_cast<const uint32_t*>(source), metadata,
        metadata == nullptr ? nullptr : metadata + rank, rank, storage_offset, count);
  } else {
    strided_copy_kernel<<<blocks, threads_per_block>>>(
        static_cast<uint64_t*>(destination), static_cast<const uint64_t*>(source), metadata,
        metadata == nullptr ? nullptr : metadata + rank, rank, storage_offset, count);
  }
  return finish_launch_and_free(metadata, previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_broadcast_reduce(
    void* destination, const void* gradient, uint64_t gradient_rank,
    const uint64_t* gradient_extents, uint64_t original_rank, const uint64_t* original_extents,
    const uint64_t* original_strides, uint64_t count, int dtype, int device) {
  if (!valid_dtype(dtype) || original_rank > gradient_rank ||
      (count != 0 && (destination == nullptr || gradient == nullptr)) ||
      (gradient_rank != 0 && gradient_extents == nullptr) ||
      (original_rank != 0 && (original_extents == nullptr || original_strides == nullptr)) ||
      original_rank > UINT64_MAX / 2U || gradient_rank > UINT64_MAX - original_rank * 2U ||
      gradient_rank + original_rank * 2U > SIZE_MAX / sizeof(uint64_t)) {
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
  const uint64_t metadata_count = gradient_rank + original_rank * 2U;
  uint64_t* metadata = nullptr;
  cudaError_t error =
      metadata_count == 0
          ? cudaSuccess
          : cudaMalloc(&metadata, static_cast<size_t>(metadata_count) * sizeof(uint64_t));
  if (error == cudaSuccess && gradient_rank != 0) {
    error =
        cudaMemcpy(metadata, gradient_extents,
                   static_cast<size_t>(gradient_rank) * sizeof(uint64_t), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess && original_rank != 0) {
    error =
        cudaMemcpy(metadata + gradient_rank, original_extents,
                   static_cast<size_t>(original_rank) * sizeof(uint64_t), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess && original_rank != 0) {
    error =
        cudaMemcpy(metadata + gradient_rank + original_rank, original_strides,
                   static_cast<size_t>(original_rank) * sizeof(uint64_t), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    return restore_after_setup_failure(error, metadata, previous, changed);
  }
  const unsigned int blocks = block_count(count);
  const uint64_t* device_original_extents =
      metadata == nullptr ? nullptr : metadata + gradient_rank;
  const uint64_t* device_original_strides =
      device_original_extents == nullptr ? nullptr : device_original_extents + original_rank;
  if (dtype == SPAR_CUDA_FLOAT32) {
    broadcast_reduce_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(destination), static_cast<const float*>(gradient), metadata,
        device_original_extents, device_original_strides, gradient_rank, original_rank, count);
  } else {
    broadcast_reduce_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(destination), static_cast<const double*>(gradient), metadata,
        device_original_extents, device_original_strides, gradient_rank, original_rank, count);
  }
  return finish_launch_and_free(metadata, previous, changed);
}

extern "C" SparCudaStatus spar_cuda_validate_targets(int32_t* invalid_target, const void* targets,
                                                     uint64_t count, uint64_t classes,
                                                     int target_dtype, int device) {
  if (invalid_target == nullptr || classes == 0 || !valid_index_dtype(target_dtype) ||
      (count != 0 && targets == nullptr)) {
    return invalid_argument();
  }
  *invalid_target = 0;
  if (count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  int32_t* device_invalid = nullptr;
  cudaError_t error = cudaMalloc(&device_invalid, sizeof(int32_t));
  if (error == cudaSuccess) {
    error = cudaMemset(device_invalid, 0, sizeof(int32_t));
  }
  if (error != cudaSuccess) {
    return restore_after_setup_failure(error, device_invalid, previous, changed);
  }
  const unsigned int blocks = block_count(count);
  if (target_dtype == SPAR_CUDA_INT32) {
    validate_targets_kernel<<<blocks, threads_per_block>>>(
        device_invalid, static_cast<const int32_t*>(targets), count, classes);
  } else {
    validate_targets_kernel<<<blocks, threads_per_block>>>(
        device_invalid, static_cast<const int64_t*>(targets), count, classes);
  }
  return finish_launch_copy_and_free(invalid_target, device_invalid, sizeof(int32_t),
                                     device_invalid, previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_nll_forward(void* output, const void* log_probabilities,
                                                       const void* targets, uint64_t output_count,
                                                       uint64_t classes, uint64_t sequence_length,
                                                       int dtype, int target_dtype,
                                                       int language_model, int device) {
  if (!valid_dtype(dtype) || !valid_index_dtype(target_dtype) || classes == 0 ||
      (language_model != 0 && language_model != 1) ||
      (language_model != 0 && sequence_length < 2) ||
      (output_count != 0 &&
       (output == nullptr || log_probabilities == nullptr || targets == nullptr))) {
    return invalid_argument();
  }
  if (output_count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(output_count);
  if (dtype == SPAR_CUDA_FLOAT32 && target_dtype == SPAR_CUDA_INT32) {
    nll_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(log_probabilities),
        static_cast<const int32_t*>(targets), output_count, classes, sequence_length,
        language_model != 0);
  } else if (dtype == SPAR_CUDA_FLOAT32) {
    nll_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(log_probabilities),
        static_cast<const int64_t*>(targets), output_count, classes, sequence_length,
        language_model != 0);
  } else if (target_dtype == SPAR_CUDA_INT32) {
    nll_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(log_probabilities),
        static_cast<const int32_t*>(targets), output_count, classes, sequence_length,
        language_model != 0);
  } else {
    nll_forward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(log_probabilities),
        static_cast<const int64_t*>(targets), output_count, classes, sequence_length,
        language_model != 0);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_nll_backward(void* output, const void* gradient,
                                                        const void* targets, uint64_t output_count,
                                                        uint64_t classes, uint64_t sequence_length,
                                                        int dtype, int target_dtype,
                                                        int language_model, int device) {
  if (!valid_dtype(dtype) || !valid_index_dtype(target_dtype) || classes == 0 ||
      (language_model != 0 && language_model != 1) ||
      (language_model != 0 && sequence_length < 2) ||
      (output_count != 0 && (output == nullptr || gradient == nullptr || targets == nullptr))) {
    return invalid_argument();
  }
  if (output_count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  const unsigned int blocks = block_count(output_count);
  if (dtype == SPAR_CUDA_FLOAT32 && target_dtype == SPAR_CUDA_INT32) {
    nll_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(gradient),
        static_cast<const int32_t*>(targets), output_count, classes, sequence_length,
        language_model != 0);
  } else if (dtype == SPAR_CUDA_FLOAT32) {
    nll_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(output), static_cast<const float*>(gradient),
        static_cast<const int64_t*>(targets), output_count, classes, sequence_length,
        language_model != 0);
  } else if (target_dtype == SPAR_CUDA_INT32) {
    nll_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(gradient),
        static_cast<const int32_t*>(targets), output_count, classes, sequence_length,
        language_model != 0);
  } else {
    nll_backward_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(output), static_cast<const double*>(gradient),
        static_cast<const int64_t*>(targets), output_count, classes, sequence_length,
        language_model != 0);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_gradient_norm_summary(SparCudaNormSummary* host_summary,
                                                          const void* gradient, uint64_t count,
                                                          int dtype, int device) {
  if (host_summary == nullptr || !valid_dtype(dtype) || (count != 0 && gradient == nullptr)) {
    return invalid_argument();
  }
  *host_summary = SparCudaNormSummary{0.0, 1.0, 0, 0};
  if (count == 0) {
    return success();
  }
  int previous = 0;
  bool changed = false;
  SparCudaStatus status = select_device(device, &previous, &changed);
  if (status.code != 0) {
    return status;
  }
  SparCudaNormSummary* device_summary = nullptr;
  const cudaError_t error = cudaMalloc(&device_summary, sizeof(SparCudaNormSummary));
  if (error != cudaSuccess) {
    return restore_after_setup_failure(error, device_summary, previous, changed);
  }
  if (dtype == SPAR_CUDA_FLOAT32) {
    gradient_norm_kernel<<<1, threads_per_block>>>(device_summary,
                                                   static_cast<const float*>(gradient), count);
  } else {
    gradient_norm_kernel<<<1, threads_per_block>>>(device_summary,
                                                   static_cast<const double*>(gradient), count);
  }
  return finish_launch_copy_and_free(host_summary, device_summary, sizeof(SparCudaNormSummary),
                                     device_summary, previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_scale_in_place(void* values, uint64_t count, int dtype,
                                                          double factor, int device) {
  if (!valid_dtype(dtype) || !isfinite(factor) || (count != 0 && values == nullptr)) {
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
    scale_in_place_kernel<<<blocks, threads_per_block>>>(static_cast<float*>(values), count,
                                                         factor);
  } else {
    scale_in_place_kernel<<<blocks, threads_per_block>>>(static_cast<double*>(values), count,
                                                         factor);
  }
  return finish_launch(previous, changed);
}

extern "C" SparCudaStatus spar_cuda_launch_adamw(void* parameter, const void* gradient,
                                                 void* first_moment, void* second_moment,
                                                 uint64_t count, int dtype, double learning_rate,
                                                 double beta1, double beta2, double epsilon,
                                                 double weight_decay, double first_correction,
                                                 double second_correction, int device) {
  if (!valid_dtype(dtype) ||
      (count != 0 && (parameter == nullptr || gradient == nullptr || first_moment == nullptr ||
                      second_moment == nullptr))) {
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
    adamw_kernel<<<blocks, threads_per_block>>>(
        static_cast<float*>(parameter), static_cast<const float*>(gradient),
        static_cast<float*>(first_moment), static_cast<float*>(second_moment), count, learning_rate,
        beta1, beta2, epsilon, weight_decay, first_correction, second_correction);
  } else {
    adamw_kernel<<<blocks, threads_per_block>>>(
        static_cast<double*>(parameter), static_cast<const double*>(gradient),
        static_cast<double*>(first_moment), static_cast<double*>(second_moment), count,
        learning_rate, beta1, beta2, epsilon, weight_decay, first_correction, second_correction);
  }
  return finish_launch(previous, changed);
}

extern "C" const char* spar_cuda_error_string(int cuda_error) {
  return cudaGetErrorString(static_cast<cudaError_t>(cuda_error));
}
