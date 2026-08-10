module;

#if SPAR_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

export module spar.cuda_runtime;

import std;

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

} // namespace spar::detail::cuda
