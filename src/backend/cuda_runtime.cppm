module;

#if SPAR_ENABLE_CUDA
#include "cuda_kernels.hpp"
#include <cuda_runtime_api.h>
#endif

export module spar.cuda_runtime;

import std;
import spar.dtype;

export namespace spar::detail::cuda {

[[nodiscard]] bool compiled() noexcept;
void validate_device(std::int32_t index);
[[nodiscard]] void* allocate(std::size_t bytes, std::int32_t device);
void deallocate(void* pointer, std::int32_t device) noexcept;
void zero(void* destination, std::int32_t destination_device, std::size_t bytes);
void copy_host_to_device(void* destination, std::int32_t destination_device, const void* source,
                         std::size_t bytes);
void copy_device_to_host(void* destination, const void* source, std::int32_t source_device,
                         std::size_t bytes);
void copy_device_to_device(void* destination, std::int32_t destination_device, const void* source,
                           std::int32_t source_device, std::size_t bytes);
void add_in_place(void* destination, const void* source, std::size_t count, DType dtype,
                  std::int32_t device);
void strided_copy(void* destination, const void* source, std::span<const std::uint64_t> extents,
                  std::span<const std::uint64_t> strides, std::uint64_t storage_offset,
                  std::size_t count, std::size_t element_size, std::int32_t device);
void broadcast_reduce(void* destination, const void* gradient,
                      std::span<const std::uint64_t> gradient_extents,
                      std::span<const std::uint64_t> original_extents,
                      std::span<const std::uint64_t> original_strides, std::size_t count,
                      DType dtype, std::int32_t device);

} // namespace spar::detail::cuda

namespace spar::detail::cuda {

namespace {

#if SPAR_ENABLE_CUDA

[[noreturn]] void throw_cuda_error(std::string_view operation, std::int32_t device,
                                   cudaError_t error) {
  throw std::runtime_error{std::string{operation} + " failed on cuda:" + std::to_string(device) +
                           ": " + cudaGetErrorString(error)};
}

void check(cudaError_t error, std::string_view operation, std::int32_t device) {
  if (error != cudaSuccess) {
    throw_cuda_error(operation, device, error);
  }
}

void check_kernel(SparCudaStatus status, std::string_view operation, std::int32_t device) {
  if (status.code != 0) {
    const char* message{spar_cuda_error_string(status.cuda_error)};
    throw std::runtime_error{std::string{operation} + " failed on cuda:" + std::to_string(device) +
                             ": " + (message == nullptr ? "unknown CUDA error" : message) +
                             " (code " + std::to_string(status.cuda_error) + ")"};
  }
}

class ScopedDevice final {
public:
  explicit ScopedDevice(std::int32_t target) : previous_{0}, changed_{false} {
    check(cudaGetDevice(&previous_), "cudaGetDevice", target);
    if (previous_ != target) {
      check(cudaSetDevice(target), "cudaSetDevice", target);
      changed_ = true;
    }
  }

  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;

  ~ScopedDevice() noexcept {
    if (changed_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }

private:
  std::int32_t previous_;
  bool changed_;
};

#else

[[noreturn]] void unavailable() {
  throw std::runtime_error{"Spar was built without CUDA support"};
}

#endif

} // namespace

bool compiled() noexcept {
#if SPAR_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

void validate_device(std::int32_t index) {
#if SPAR_ENABLE_CUDA
  int count{0};
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", index);
  if (index < 0 || index >= count) {
    throw std::invalid_argument{"CUDA device index is unavailable: cuda:" + std::to_string(index)};
  }
#else
  static_cast<void>(index);
  unavailable();
#endif
}

void* allocate(std::size_t bytes, std::int32_t device) {
#if SPAR_ENABLE_CUDA
  validate_device(device);
  ScopedDevice guard{device};
  if (bytes == 0) {
    return nullptr;
  }
  void* pointer{nullptr};
  check(cudaMalloc(&pointer, bytes), "cudaMalloc", device);
  return pointer;
#else
  static_cast<void>(bytes);
  static_cast<void>(device);
  unavailable();
#endif
}

void deallocate(void* pointer, std::int32_t device) noexcept {
#if SPAR_ENABLE_CUDA
  if (pointer == nullptr) {
    return;
  }
  int previous{0};
  if (cudaGetDevice(&previous) != cudaSuccess) {
    return;
  }
  const bool changed{previous != device};
  if (changed && cudaSetDevice(device) != cudaSuccess) {
    return;
  }
  static_cast<void>(cudaFree(pointer));
  if (changed) {
    static_cast<void>(cudaSetDevice(previous));
  }
#else
  static_cast<void>(pointer);
  static_cast<void>(device);
#endif
}

void zero(void* destination, std::int32_t destination_device, std::size_t bytes) {
#if SPAR_ENABLE_CUDA
  validate_device(destination_device);
  ScopedDevice guard{destination_device};
  if (bytes != 0) {
    check(cudaMemset(destination, 0, bytes), "cudaMemset", destination_device);
  }
#else
  static_cast<void>(destination);
  static_cast<void>(destination_device);
  static_cast<void>(bytes);
  unavailable();
#endif
}

void copy_host_to_device(void* destination, std::int32_t destination_device, const void* source,
                         std::size_t bytes) {
#if SPAR_ENABLE_CUDA
  validate_device(destination_device);
  ScopedDevice guard{destination_device};
  if (bytes != 0) {
    check(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D",
          destination_device);
  }
#else
  static_cast<void>(destination);
  static_cast<void>(destination_device);
  static_cast<void>(source);
  static_cast<void>(bytes);
  unavailable();
#endif
}

void copy_device_to_host(void* destination, const void* source, std::int32_t source_device,
                         std::size_t bytes) {
#if SPAR_ENABLE_CUDA
  validate_device(source_device);
  ScopedDevice guard{source_device};
  if (bytes != 0) {
    check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H",
          source_device);
  }
#else
  static_cast<void>(destination);
  static_cast<void>(source);
  static_cast<void>(source_device);
  static_cast<void>(bytes);
  unavailable();
#endif
}

void copy_device_to_device(void* destination, std::int32_t destination_device, const void* source,
                           std::int32_t source_device, std::size_t bytes) {
#if SPAR_ENABLE_CUDA
  validate_device(destination_device);
  validate_device(source_device);
  if (bytes == 0) {
    return;
  }
  if (destination_device == source_device) {
    ScopedDevice guard{destination_device};
    check(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D",
          destination_device);
    return;
  }
  std::vector<std::byte> staging(bytes);
  copy_device_to_host(staging.data(), source, source_device, bytes);
  copy_host_to_device(destination, destination_device, staging.data(), bytes);
#else
  static_cast<void>(destination);
  static_cast<void>(destination_device);
  static_cast<void>(source);
  static_cast<void>(source_device);
  static_cast<void>(bytes);
  unavailable();
#endif
}

void add_in_place(void* destination, const void* source, std::size_t count, DType dtype,
                  std::int32_t device) {
#if SPAR_ENABLE_CUDA
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (count > std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error{"CUDA gradient element count exceeds uint64_t"};
    }
  }
  int dtype_tag{0};
  switch (dtype) {
  case DType::Float32:
    dtype_tag = SPAR_CUDA_FLOAT32;
    break;
  case DType::Float64:
    dtype_tag = SPAR_CUDA_FLOAT64;
    break;
  case DType::Int32:
  case DType::Int64:
    throw std::logic_error{"CUDA gradient accumulation requires a floating dtype"};
  }
  check_kernel(spar_cuda_launch_add_in_place(destination, source, static_cast<std::uint64_t>(count),
                                             dtype_tag, device),
               "CUDA gradient accumulation", device);
#else
  static_cast<void>(destination);
  static_cast<void>(source);
  static_cast<void>(count);
  static_cast<void>(dtype);
  static_cast<void>(device);
  unavailable();
#endif
}

void strided_copy(void* destination, const void* source, std::span<const std::uint64_t> extents,
                  std::span<const std::uint64_t> strides, std::uint64_t storage_offset,
                  std::size_t count, std::size_t element_size, std::int32_t device) {
#if SPAR_ENABLE_CUDA
  if (extents.size() != strides.size()) {
    throw std::invalid_argument{"CUDA strided copy metadata ranks differ"};
  }
  if (extents.size() > std::numeric_limits<std::uint64_t>::max() ||
      count > std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"CUDA strided copy metadata exceeds uint64_t"};
  }
  check_kernel(spar_cuda_launch_strided_copy(
                   destination, source, static_cast<std::uint64_t>(extents.size()), extents.data(),
                   strides.data(), storage_offset, static_cast<std::uint64_t>(count),
                   static_cast<std::uint64_t>(element_size), device),
               "CUDA strided copy", device);
#else
  static_cast<void>(destination);
  static_cast<void>(source);
  static_cast<void>(extents);
  static_cast<void>(strides);
  static_cast<void>(storage_offset);
  static_cast<void>(count);
  static_cast<void>(element_size);
  static_cast<void>(device);
  unavailable();
#endif
}

void broadcast_reduce(void* destination, const void* gradient,
                      std::span<const std::uint64_t> gradient_extents,
                      std::span<const std::uint64_t> original_extents,
                      std::span<const std::uint64_t> original_strides, std::size_t count,
                      DType dtype, std::int32_t device) {
#if SPAR_ENABLE_CUDA
  if (original_extents.size() != original_strides.size() ||
      original_extents.size() > gradient_extents.size()) {
    throw std::invalid_argument{"CUDA broadcast reduction metadata is inconsistent"};
  }
  if (gradient_extents.size() > std::numeric_limits<std::uint64_t>::max() ||
      original_extents.size() > std::numeric_limits<std::uint64_t>::max() ||
      count > std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error{"CUDA broadcast reduction metadata exceeds uint64_t"};
  }
  int dtype_tag{0};
  switch (dtype) {
  case DType::Float32:
    dtype_tag = SPAR_CUDA_FLOAT32;
    break;
  case DType::Float64:
    dtype_tag = SPAR_CUDA_FLOAT64;
    break;
  case DType::Int32:
  case DType::Int64:
    throw std::logic_error{"CUDA broadcast-gradient reduction requires a floating dtype"};
  }
  check_kernel(spar_cuda_launch_broadcast_reduce(
                   destination, gradient, static_cast<std::uint64_t>(gradient_extents.size()),
                   gradient_extents.data(), static_cast<std::uint64_t>(original_extents.size()),
                   original_extents.data(), original_strides.data(),
                   static_cast<std::uint64_t>(count), dtype_tag, device),
               "CUDA broadcast-gradient reduction", device);
#else
  static_cast<void>(destination);
  static_cast<void>(gradient);
  static_cast<void>(gradient_extents);
  static_cast<void>(original_extents);
  static_cast<void>(original_strides);
  static_cast<void>(count);
  static_cast<void>(dtype);
  static_cast<void>(device);
  unavailable();
#endif
}

} // namespace spar::detail::cuda
