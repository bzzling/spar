module;

#if SPAR_ENABLE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

export module spar.cuda_blas;

import std;
import spar.dtype;
import spar.shape;
import spar.tensor;

export namespace spar::detail::cuda_blas {

[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b, Shape output_shape, std::size_t m,
                            std::size_t k, std::size_t n);

} // namespace spar::detail::cuda_blas

namespace spar::detail::cuda_blas {
namespace {

#if SPAR_ENABLE_CUDA

[[noreturn]] void throw_cuda(std::string_view operation, Device device, cudaError_t status) {
  throw std::runtime_error{std::string{operation} + " failed on cuda:" +
                           std::to_string(device.index()) + ": " + cudaGetErrorString(status)};
}

void check_cuda(cudaError_t status, std::string_view operation, Device device) {
  if (status != cudaSuccess) {
    throw_cuda(operation, device, status);
  }
}

[[noreturn]] void throw_cublas(std::string_view operation, Device device, cublasStatus_t status) {
  throw std::runtime_error{std::string{operation} +
                           " failed on cuda:" + std::to_string(device.index()) +
                           ": cuBLAS status " + std::to_string(static_cast<int>(status))};
}

void check_cublas(cublasStatus_t status, std::string_view operation, Device device) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw_cublas(operation, device, status);
  }
}

class ScopedDevice final {
public:
  explicit ScopedDevice(Device target) : target_{target}, previous_{0}, changed_{false} {
    check_cuda(cudaGetDevice(&previous_), "cudaGetDevice for matmul", target_);
    if (previous_ != target_.index()) {
      check_cuda(cudaSetDevice(target_.index()), "cudaSetDevice for matmul", target_);
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
  Device target_;
  int previous_;
  bool changed_;
};

class Handle final {
public:
  explicit Handle(Device device) : value_{nullptr} {
    check_cublas(cublasCreate(&value_), "cublasCreate for matmul", device);
  }

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;

  ~Handle() noexcept {
    if (value_ != nullptr) {
      static_cast<void>(cublasDestroy(value_));
    }
  }

  [[nodiscard]] cublasHandle_t get() const noexcept {
    return value_;
  }

private:
  cublasHandle_t value_;
};

int checked_dimension(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error{"cuBLAS matrix dimension exceeds int"};
  }
  return static_cast<int>(value);
}

std::size_t checked_product(std::size_t left, std::size_t right, std::string_view description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error{std::string{description} + " overflow"};
  }
  return left * right;
}

#else

[[noreturn]] void unavailable() {
  throw std::runtime_error{"Spar was built without CUDA support"};
}

#endif

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b, Shape output_shape, std::size_t m, std::size_t k,
              std::size_t n) {
#if SPAR_ENABLE_CUDA
  if (!a.device().is_cuda() || a.device() != b.device() || !a.is_contiguous() ||
      !b.is_contiguous() || a.dtype() != b.dtype() ||
      (a.dtype() != DType::Float32 && a.dtype() != DType::Float64)) {
    throw std::logic_error{"Internal CUDA matmul received invalid tensors"};
  }
  Tensor output{std::move(output_shape), a.dtype(), a.device()};
  if (output.numel() == 0) {
    return output;
  }
  if (k == 0) {
    return zeros(output.shape(), output.dtype(), output.device());
  }
  if (m == 0 || n == 0) {
    throw std::logic_error{"Internal CUDA matmul received invalid geometry"};
  }
  const std::size_t matrix_output_size{checked_product(m, n, "CUDA matmul output matrix size")};
  if (output.numel() % matrix_output_size != 0) {
    throw std::logic_error{"Internal CUDA matmul output geometry is inconsistent"};
  }
  const std::size_t batch_count{output.numel() / matrix_output_size};
  const int cublas_m{checked_dimension(n)};
  const int cublas_n{checked_dimension(m)};
  const int cublas_k{checked_dimension(k)};
  const std::size_t a_matrix_size{checked_product(m, k, "CUDA matmul left matrix size")};
  const std::size_t b_matrix_size{checked_product(k, n, "CUDA matmul right matrix size")};
  if (a.numel() != checked_product(batch_count, a_matrix_size, "CUDA matmul left batch size") ||
      b.numel() != checked_product(batch_count, b_matrix_size, "CUDA matmul right batch size")) {
    throw std::logic_error{"Internal CUDA matmul input geometry is inconsistent"};
  }
  ScopedDevice guard{output.device()};
  Handle handle{output.device()};
  if (output.dtype() == DType::Float32) {
    const float alpha{1.0F};
    const float beta{0.0F};
    const auto* a_values{static_cast<const float*>(CudaTensorAccess::data(a))};
    const auto* b_values{static_cast<const float*>(CudaTensorAccess::data(b))};
    auto* output_values{static_cast<float*>(CudaTensorAccess::mutable_data(output))};
    for (std::size_t batch{0}; batch < batch_count; ++batch) {
      check_cublas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, cublas_m, cublas_n, cublas_k,
                               &alpha, b_values + batch * b_matrix_size, cublas_m,
                               a_values + batch * a_matrix_size, cublas_k, &beta,
                               output_values + batch * matrix_output_size, cublas_m),
                   "cublasSgemm", output.device());
    }
  } else {
    const double alpha{1.0};
    const double beta{0.0};
    const auto* a_values{static_cast<const double*>(CudaTensorAccess::data(a))};
    const auto* b_values{static_cast<const double*>(CudaTensorAccess::data(b))};
    auto* output_values{static_cast<double*>(CudaTensorAccess::mutable_data(output))};
    for (std::size_t batch{0}; batch < batch_count; ++batch) {
      check_cublas(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, cublas_m, cublas_n, cublas_k,
                               &alpha, b_values + batch * b_matrix_size, cublas_m,
                               a_values + batch * a_matrix_size, cublas_k, &beta,
                               output_values + batch * matrix_output_size, cublas_m),
                   "cublasDgemm", output.device());
    }
  }
  check_cuda(cudaDeviceSynchronize(), "CUDA matmul synchronization", output.device());
  return output;
#else
  static_cast<void>(a);
  static_cast<void>(b);
  static_cast<void>(output_shape);
  static_cast<void>(m);
  static_cast<void>(k);
  static_cast<void>(n);
  unavailable();
#endif
}

} // namespace spar::detail::cuda_blas
