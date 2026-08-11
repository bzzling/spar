module;

#if SPAR_ENABLE_CUDA
#include "cuda_kernels.hpp"
#endif

export module spar.cuda_ops;

import std;
import spar.device;
import spar.dtype;
import spar.shape;
import spar.tensor;

export namespace spar::detail::cuda_ops {

enum class BinaryOperation { Add, Subtract, Multiply, Divide };
enum class UnaryOperation { Negate, Square, Reciprocal, Exp, Log, Sqrt, Sigmoid, Silu };
enum class ReductionOperation { Sum, Mean, Max };

[[nodiscard]] Tensor binary(const Tensor& left, const Tensor& right, BinaryOperation operation);
[[nodiscard]] Tensor scalar(const Tensor& input, double value, BinaryOperation operation);
[[nodiscard]] Tensor unary(const Tensor& input, UnaryOperation operation);
[[nodiscard]] Tensor reduction(const Tensor& input, ReductionOperation operation);
[[nodiscard]] Tensor fill_from_device_scalar(Shape shape, const Tensor& scalar, double scale);

} // namespace spar::detail::cuda_ops

namespace spar::detail::cuda_ops {
namespace {

#if SPAR_ENABLE_CUDA

int dtype_tag(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return SPAR_CUDA_FLOAT32;
  case DType::Float64:
    return SPAR_CUDA_FLOAT64;
  case DType::Int32:
  case DType::Int64:
    throw std::logic_error{"CUDA operation dtype validation invariant violated"};
  }
  throw std::logic_error{"CUDA operation dtype validation invariant violated"};
}

int binary_tag(BinaryOperation operation) {
  switch (operation) {
  case BinaryOperation::Add:
    return SPAR_CUDA_ADD;
  case BinaryOperation::Subtract:
    return SPAR_CUDA_SUBTRACT;
  case BinaryOperation::Multiply:
    return SPAR_CUDA_MULTIPLY;
  case BinaryOperation::Divide:
    return SPAR_CUDA_DIVIDE;
  }
  throw std::logic_error{"Unknown CUDA binary operation"};
}

int unary_tag(UnaryOperation operation) {
  switch (operation) {
  case UnaryOperation::Negate:
    return SPAR_CUDA_NEGATE;
  case UnaryOperation::Square:
    return SPAR_CUDA_SQUARE;
  case UnaryOperation::Reciprocal:
    return SPAR_CUDA_RECIPROCAL;
  case UnaryOperation::Exp:
    return SPAR_CUDA_EXP;
  case UnaryOperation::Log:
    return SPAR_CUDA_LOG;
  case UnaryOperation::Sqrt:
    return SPAR_CUDA_SQRT;
  case UnaryOperation::Sigmoid:
    return SPAR_CUDA_SIGMOID;
  case UnaryOperation::Silu:
    return SPAR_CUDA_SILU;
  }
  throw std::logic_error{"Unknown CUDA unary operation"};
}

int reduction_tag(ReductionOperation operation) {
  switch (operation) {
  case ReductionOperation::Sum:
    return SPAR_CUDA_SUM;
  case ReductionOperation::Mean:
    return SPAR_CUDA_MEAN;
  case ReductionOperation::Max:
    return SPAR_CUDA_MAX;
  }
  throw std::logic_error{"Unknown CUDA reduction operation"};
}

std::string_view name_of(BinaryOperation operation) {
  switch (operation) {
  case BinaryOperation::Add:
    return "add";
  case BinaryOperation::Subtract:
    return "subtract";
  case BinaryOperation::Multiply:
    return "multiply";
  case BinaryOperation::Divide:
    return "divide";
  }
  return "binary operation";
}

std::string_view name_of(UnaryOperation operation) {
  switch (operation) {
  case UnaryOperation::Negate:
    return "negate";
  case UnaryOperation::Square:
    return "square";
  case UnaryOperation::Reciprocal:
    return "reciprocal";
  case UnaryOperation::Exp:
    return "exp";
  case UnaryOperation::Log:
    return "log";
  case UnaryOperation::Sqrt:
    return "sqrt";
  case UnaryOperation::Sigmoid:
    return "sigmoid";
  case UnaryOperation::Silu:
    return "silu";
  }
  return "unary operation";
}

std::string_view name_of(ReductionOperation operation) {
  switch (operation) {
  case ReductionOperation::Sum:
    return "sum";
  case ReductionOperation::Mean:
    return "mean";
  case ReductionOperation::Max:
    return "reduce_max";
  }
  return "reduction";
}

[[noreturn]] void throw_status(std::string_view operation, Device device, SparCudaStatus status) {
  const char* message{spar_cuda_error_string(status.cuda_error)};
  throw std::runtime_error{std::string{operation} +
                           " failed on cuda:" + std::to_string(device.index()) + ": " +
                           (message == nullptr ? "unknown CUDA error" : message) + " (code " +
                           std::to_string(status.cuda_error) + ")"};
}

void check(SparCudaStatus status, std::string_view operation, Device device) {
  if (status.code != 0) {
    throw_status(operation, device, status);
  }
}

std::uint64_t checked_count(std::size_t count) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (count > std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error{"CUDA logical element count exceeds uint64_t"};
    }
  }
  return static_cast<std::uint64_t>(count);
}

#else

[[noreturn]] void unavailable() {
  throw std::runtime_error{"Spar was built without CUDA support"};
}

#endif

} // namespace

Tensor binary(const Tensor& left, const Tensor& right, BinaryOperation operation) {
#if SPAR_ENABLE_CUDA
  if (!left.device().is_cuda() || left.device() != right.device() ||
      left.shape() != right.shape() || left.dtype() != right.dtype() || !left.is_contiguous() ||
      !right.is_contiguous()) {
    throw std::logic_error{"Internal CUDA binary operation received invalid tensors"};
  }
  Tensor output{left.shape(), left.dtype(), left.device()};
  check(spar_cuda_launch_binary(CudaTensorAccess::mutable_data(output),
                                CudaTensorAccess::data(left), CudaTensorAccess::data(right),
                                checked_count(output.numel()), dtype_tag(output.dtype()),
                                binary_tag(operation), output.device().index()),
        name_of(operation), output.device());
  return output;
#else
  static_cast<void>(left);
  static_cast<void>(right);
  static_cast<void>(operation);
  unavailable();
#endif
}

Tensor scalar(const Tensor& input, double value, BinaryOperation operation) {
#if SPAR_ENABLE_CUDA
  if (!input.device().is_cuda() || !input.is_contiguous()) {
    throw std::logic_error{"Internal CUDA scalar operation received an invalid Tensor"};
  }
  Tensor output{input.shape(), input.dtype(), input.device()};
  check(spar_cuda_launch_scalar(CudaTensorAccess::mutable_data(output),
                                CudaTensorAccess::data(input), checked_count(output.numel()),
                                dtype_tag(output.dtype()), binary_tag(operation), value,
                                output.device().index()),
        name_of(operation), output.device());
  return output;
#else
  static_cast<void>(input);
  static_cast<void>(value);
  static_cast<void>(operation);
  unavailable();
#endif
}

Tensor unary(const Tensor& input, UnaryOperation operation) {
#if SPAR_ENABLE_CUDA
  if (!input.device().is_cuda() || !input.is_contiguous()) {
    throw std::logic_error{"Internal CUDA unary operation received an invalid Tensor"};
  }
  Tensor output{input.shape(), input.dtype(), input.device()};
  check(spar_cuda_launch_unary(CudaTensorAccess::mutable_data(output),
                               CudaTensorAccess::data(input), checked_count(output.numel()),
                               dtype_tag(output.dtype()), unary_tag(operation),
                               output.device().index()),
        name_of(operation), output.device());
  return output;
#else
  static_cast<void>(input);
  static_cast<void>(operation);
  unavailable();
#endif
}

Tensor reduction(const Tensor& input, ReductionOperation operation) {
#if SPAR_ENABLE_CUDA
  if (!input.device().is_cuda() || !input.is_contiguous()) {
    throw std::logic_error{"Internal CUDA reduction received an invalid Tensor"};
  }
  Tensor output{Shape{}, input.dtype(), input.device()};
  check(spar_cuda_launch_reduction(CudaTensorAccess::mutable_data(output),
                                   CudaTensorAccess::data(input), checked_count(input.numel()),
                                   dtype_tag(input.dtype()), reduction_tag(operation),
                                   input.device().index()),
        name_of(operation), input.device());
  return output;
#else
  static_cast<void>(input);
  static_cast<void>(operation);
  unavailable();
#endif
}

Tensor fill_from_device_scalar(Shape shape, const Tensor& scalar, double scale) {
#if SPAR_ENABLE_CUDA
  if (!scalar.device().is_cuda() || !scalar.is_contiguous() || scalar.numel() != 1) {
    throw std::logic_error{"Internal CUDA scalar fill received an invalid scalar Tensor"};
  }
  Tensor output{std::move(shape), scalar.dtype(), scalar.device()};
  check(spar_cuda_launch_fill_from_scalar(CudaTensorAccess::mutable_data(output),
                                          CudaTensorAccess::data(scalar),
                                          checked_count(output.numel()), dtype_tag(output.dtype()),
                                          scale, output.device().index()),
        "fill from device scalar", output.device());
  return output;
#else
  static_cast<void>(shape);
  static_cast<void>(scalar);
  static_cast<void>(scale);
  unavailable();
#endif
}

} // namespace spar::detail::cuda_ops
