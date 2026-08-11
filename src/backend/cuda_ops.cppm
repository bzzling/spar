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

struct SoftmaxForwardResult final {
  Tensor output;
  Tensor undefined_slices;
};

[[nodiscard]] Tensor binary(const Tensor& left, const Tensor& right, BinaryOperation operation);
[[nodiscard]] Tensor scalar(const Tensor& input, double value, BinaryOperation operation);
[[nodiscard]] Tensor unary(const Tensor& input, UnaryOperation operation);
[[nodiscard]] Tensor reduction(const Tensor& input, ReductionOperation operation);
[[nodiscard]] Tensor row_reduction(const Tensor& input, std::size_t row_count,
                                   std::size_t reduction_count, ReductionOperation operation);
[[nodiscard]] SoftmaxForwardResult probability_forward(const Tensor& input, std::size_t outer,
                                                       std::size_t axis_extent, std::size_t inner,
                                                       bool logarithmic);
[[nodiscard]] Tensor probability_backward(const Tensor& gradient, const Tensor& saved_output,
                                          const Tensor& undefined_slices, std::size_t outer,
                                          std::size_t axis_extent, std::size_t inner,
                                          bool logarithmic);
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

Tensor row_reduction(const Tensor& input, std::size_t row_count, std::size_t reduction_count,
                     ReductionOperation operation) {
#if SPAR_ENABLE_CUDA
  const bool product_matches{reduction_count == 0
                                 ? input.numel() == 0
                                 : row_count <= std::numeric_limits<std::size_t>::max() /
                                                    reduction_count &&
                                       input.numel() == row_count * reduction_count};
  if (!input.device().is_cuda() || !input.is_contiguous() || operation == ReductionOperation::Max ||
      !product_matches ||
      row_count > static_cast<std::size_t>(std::numeric_limits<Shape::dimension_type>::max())) {
    throw std::logic_error{"Internal CUDA row reduction received invalid geometry"};
  }
  Tensor output{Shape{static_cast<Shape::dimension_type>(row_count)}, input.dtype(),
                input.device()};
  check(spar_cuda_launch_row_reduction(CudaTensorAccess::mutable_data(output),
                                       CudaTensorAccess::data(input), checked_count(row_count),
                                       checked_count(reduction_count), dtype_tag(input.dtype()),
                                       reduction_tag(operation), input.device().index()),
        name_of(operation), input.device());
  return output;
#else
  static_cast<void>(input);
  static_cast<void>(row_count);
  static_cast<void>(reduction_count);
  static_cast<void>(operation);
  unavailable();
#endif
}

SoftmaxForwardResult probability_forward(const Tensor& input, std::size_t outer,
                                         std::size_t axis_extent, std::size_t inner,
                                         bool logarithmic) {
#if SPAR_ENABLE_CUDA
  const bool geometry_fits{outer != 0 && axis_extent != 0 && inner != 0 &&
                           outer <= std::numeric_limits<std::size_t>::max() / axis_extent &&
                           outer * axis_extent <= std::numeric_limits<std::size_t>::max() / inner &&
                           input.numel() == outer * axis_extent * inner &&
                           outer <= std::numeric_limits<std::size_t>::max() / inner};
  const std::size_t slice_count{geometry_fits ? outer * inner : 0};
  if (!input.device().is_cuda() || !input.is_contiguous() || !geometry_fits ||
      slice_count > static_cast<std::size_t>(std::numeric_limits<Shape::dimension_type>::max())) {
    throw std::logic_error{"Internal CUDA probability forward received invalid geometry"};
  }
  Tensor output{input.shape(), input.dtype(), input.device()};
  Tensor undefined_slices{Shape{static_cast<Shape::dimension_type>(slice_count)}, DType::Int32,
                          input.device()};
  check(spar_cuda_launch_probability_forward(
            CudaTensorAccess::mutable_data(output),
            static_cast<std::int32_t*>(CudaTensorAccess::mutable_data(undefined_slices)),
            CudaTensorAccess::data(input), checked_count(outer), checked_count(axis_extent),
            checked_count(inner), dtype_tag(input.dtype()), logarithmic ? 1 : 0,
            input.device().index()),
        logarithmic ? "log_softmax" : "softmax", input.device());
  return {std::move(output), std::move(undefined_slices)};
#else
  static_cast<void>(input);
  static_cast<void>(outer);
  static_cast<void>(axis_extent);
  static_cast<void>(inner);
  static_cast<void>(logarithmic);
  unavailable();
#endif
}

Tensor probability_backward(const Tensor& gradient, const Tensor& saved_output,
                            const Tensor& undefined_slices, std::size_t outer,
                            std::size_t axis_extent, std::size_t inner, bool logarithmic) {
#if SPAR_ENABLE_CUDA
  const bool geometry_fits{outer != 0 && axis_extent != 0 && inner != 0 &&
                           outer <= std::numeric_limits<std::size_t>::max() / axis_extent &&
                           outer * axis_extent <= std::numeric_limits<std::size_t>::max() / inner &&
                           gradient.numel() == outer * axis_extent * inner &&
                           outer <= std::numeric_limits<std::size_t>::max() / inner};
  const std::size_t slice_count{geometry_fits ? outer * inner : 0};
  if (!gradient.device().is_cuda() || gradient.device() != saved_output.device() ||
      gradient.device() != undefined_slices.device() || !gradient.is_contiguous() ||
      !saved_output.is_contiguous() || !undefined_slices.is_contiguous() ||
      gradient.shape() != saved_output.shape() || gradient.dtype() != saved_output.dtype() ||
      undefined_slices.dtype() != DType::Int32 || !geometry_fits ||
      undefined_slices.numel() != slice_count) {
    throw std::logic_error{"Internal CUDA probability backward received invalid tensors"};
  }
  Tensor output{gradient.shape(), gradient.dtype(), gradient.device()};
  check(spar_cuda_launch_probability_backward(
            CudaTensorAccess::mutable_data(output), CudaTensorAccess::data(gradient),
            CudaTensorAccess::data(saved_output),
            static_cast<const std::int32_t*>(CudaTensorAccess::data(undefined_slices)),
            checked_count(outer), checked_count(axis_extent), checked_count(inner),
            dtype_tag(output.dtype()), logarithmic ? 1 : 0, output.device().index()),
        logarithmic ? "log_softmax backward" : "softmax backward", output.device());
  return output;
#else
  static_cast<void>(gradient);
  static_cast<void>(saved_output);
  static_cast<void>(undefined_slices);
  static_cast<void>(outer);
  static_cast<void>(axis_extent);
  static_cast<void>(inner);
  static_cast<void>(logarithmic);
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
