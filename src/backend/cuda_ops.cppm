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

struct GradientNormSummary final {
  double scale;
  double scaled_sum_squares;
  bool has_infinity;
  bool has_nan;
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
[[nodiscard]] Tensor embedding_forward(const Tensor& weight, const Tensor& indices,
                                       Shape output_shape);
[[nodiscard]] Tensor embedding_backward(const Tensor& gradient, const Tensor& indices,
                                        Shape weight_shape);
[[nodiscard]] Tensor rope(const Tensor& input, std::size_t start_position, double theta,
                          bool inverse);
[[nodiscard]] Tensor causal_mask(std::size_t sequence_length, DType dtype, Device device);
[[nodiscard]] Tensor fill_from_device_scalar(Shape shape, const Tensor& scalar, double scale);
void validate_targets(const Tensor& targets, std::size_t classes);
[[nodiscard]] Tensor nll_forward(const Tensor& log_probabilities, const Tensor& targets,
                                 Shape output_shape, std::size_t sequence_length = 0);
[[nodiscard]] Tensor nll_backward(const Tensor& gradient, const Tensor& targets,
                                  Shape log_probability_shape, std::size_t sequence_length = 0);
[[nodiscard]] GradientNormSummary gradient_norm_summary(const Tensor& gradient);
void scale_in_place(Tensor& values, double factor);
void adamw_update(Tensor& parameter, const Tensor& gradient, Tensor& first_moment,
                  Tensor& second_moment, std::uint64_t step, double learning_rate, double beta1,
                  double beta2, double epsilon, double weight_decay);

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

int index_dtype_tag(DType dtype) {
  switch (dtype) {
  case DType::Int32:
    return SPAR_CUDA_INT32;
  case DType::Int64:
    return SPAR_CUDA_INT64;
  case DType::Float32:
  case DType::Float64:
    throw std::logic_error{"CUDA index dtype validation invariant violated"};
  }
  throw std::logic_error{"CUDA index dtype validation invariant violated"};
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

Tensor embedding_forward(const Tensor& weight, const Tensor& indices, Shape output_shape) {
#if SPAR_ENABLE_CUDA
  if (!weight.device().is_cuda() || weight.device() != indices.device() ||
      !weight.is_contiguous() || !indices.is_contiguous() || weight.rank() != 2 ||
      (indices.dtype() != DType::Int32 && indices.dtype() != DType::Int64)) {
    throw std::logic_error{"Internal CUDA embedding forward received invalid tensors"};
  }
  Tensor output{std::move(output_shape), weight.dtype(), weight.device()};
  if (indices.numel() == 0) {
    return output;
  }
  Tensor invalid_index{zeros(Shape{}, DType::Int32, weight.device())};
  const std::size_t vocabulary_size{static_cast<std::size_t>(weight.shape()[0])};
  const std::size_t embedding_dimension{static_cast<std::size_t>(weight.shape()[1])};
  if (embedding_dimension == 0) {
    check(spar_cuda_launch_validate_embedding_indices(
              static_cast<std::int32_t*>(CudaTensorAccess::mutable_data(invalid_index)),
              CudaTensorAccess::data(indices), checked_count(indices.numel()),
              checked_count(vocabulary_size), index_dtype_tag(indices.dtype()),
              weight.device().index()),
          "embedding_lookup index validation", weight.device());
  } else {
    check(spar_cuda_launch_embedding_forward(
              CudaTensorAccess::mutable_data(output),
              static_cast<std::int32_t*>(CudaTensorAccess::mutable_data(invalid_index)),
              CudaTensorAccess::data(weight), CudaTensorAccess::data(indices),
              checked_count(indices.numel()), checked_count(vocabulary_size),
              checked_count(embedding_dimension), dtype_tag(weight.dtype()),
              index_dtype_tag(indices.dtype()), weight.device().index()),
          "embedding_lookup", weight.device());
  }
  const Tensor host_status{invalid_index.to(Device::cpu())};
  if (host_status.span<std::int32_t>()[0] != 0) {
    throw std::out_of_range{"embedding_lookup index is outside the weight vocabulary"};
  }
  return output;
#else
  static_cast<void>(weight);
  static_cast<void>(indices);
  static_cast<void>(output_shape);
  unavailable();
#endif
}

Tensor embedding_backward(const Tensor& gradient, const Tensor& indices, Shape weight_shape) {
#if SPAR_ENABLE_CUDA
  if (!gradient.device().is_cuda() || gradient.device() != indices.device() ||
      !gradient.is_contiguous() || !indices.is_contiguous() ||
      (indices.dtype() != DType::Int32 && indices.dtype() != DType::Int64) ||
      weight_shape.rank() != 2) {
    throw std::logic_error{"Internal CUDA embedding backward received invalid tensors"};
  }
  Tensor output{zeros(weight_shape, gradient.dtype(), gradient.device())};
  if (output.numel() == 0) {
    return output;
  }
  check(spar_cuda_launch_embedding_backward(
            CudaTensorAccess::mutable_data(output), CudaTensorAccess::data(gradient),
            CudaTensorAccess::data(indices), checked_count(indices.numel()),
            checked_count(static_cast<std::size_t>(weight_shape[1])), dtype_tag(gradient.dtype()),
            index_dtype_tag(indices.dtype()), gradient.device().index()),
        "embedding backward", gradient.device());
  return output;
#else
  static_cast<void>(gradient);
  static_cast<void>(indices);
  static_cast<void>(weight_shape);
  unavailable();
#endif
}

Tensor rope(const Tensor& input, std::size_t start_position, double theta, bool inverse) {
#if SPAR_ENABLE_CUDA
  if (!input.device().is_cuda() || !input.is_contiguous() || input.rank() < 2) {
    throw std::logic_error{"Internal CUDA RoPE received an invalid Tensor"};
  }
  Tensor output{input.shape(), input.dtype(), input.device()};
  if (input.numel() == 0) {
    return output;
  }
  const std::size_t sequence_length{static_cast<std::size_t>(input.shape()[input.rank() - 2])};
  const std::size_t feature_count{static_cast<std::size_t>(input.shape()[input.rank() - 1])};
  check(spar_cuda_launch_rope(CudaTensorAccess::mutable_data(output), CudaTensorAccess::data(input),
                              checked_count(input.numel() / 2), checked_count(sequence_length),
                              checked_count(feature_count), checked_count(start_position), theta,
                              dtype_tag(input.dtype()), inverse ? 1 : 0, input.device().index()),
        inverse ? "RoPE backward" : "RoPE", input.device());
  return output;
#else
  static_cast<void>(input);
  static_cast<void>(start_position);
  static_cast<void>(theta);
  static_cast<void>(inverse);
  unavailable();
#endif
}

Tensor causal_mask(std::size_t sequence_length, DType dtype, Device device) {
#if SPAR_ENABLE_CUDA
  if (!device.is_cuda() || sequence_length == 0 ||
      sequence_length >
          static_cast<std::size_t>(std::numeric_limits<Shape::dimension_type>::max())) {
    throw std::logic_error{"Internal CUDA causal mask received invalid geometry"};
  }
  const auto dimension{static_cast<Shape::dimension_type>(sequence_length)};
  Tensor output{Shape{dimension, dimension}, dtype, device};
  check(spar_cuda_launch_causal_mask(CudaTensorAccess::mutable_data(output),
                                     checked_count(sequence_length), dtype_tag(dtype),
                                     device.index()),
        "causal attention mask", device);
  return output;
#else
  static_cast<void>(sequence_length);
  static_cast<void>(dtype);
  static_cast<void>(device);
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

void validate_targets(const Tensor& targets, std::size_t classes) {
#if SPAR_ENABLE_CUDA
  if (!targets.device().is_cuda() || !targets.is_contiguous() ||
      (targets.dtype() != DType::Int32 && targets.dtype() != DType::Int64) || classes == 0) {
    throw std::logic_error{"Internal CUDA target validation received an invalid Tensor"};
  }
  std::int32_t invalid_target{0};
  check(spar_cuda_validate_targets(&invalid_target, CudaTensorAccess::data(targets),
                                   checked_count(targets.numel()), checked_count(classes),
                                   index_dtype_tag(targets.dtype()), targets.device().index()),
        "cross_entropy target validation", targets.device());
  if (invalid_target != 0) {
    throw std::out_of_range{"cross_entropy target is outside the class range"};
  }
#else
  static_cast<void>(targets);
  static_cast<void>(classes);
  unavailable();
#endif
}

Tensor nll_forward(const Tensor& log_probabilities, const Tensor& targets, Shape output_shape,
                   std::size_t sequence_length) {
#if SPAR_ENABLE_CUDA
  if (!log_probabilities.device().is_cuda() || log_probabilities.device() != targets.device() ||
      !log_probabilities.is_contiguous() || !targets.is_contiguous() ||
      log_probabilities.rank() == 0 || output_shape.numel() == 0) {
    throw std::logic_error{"Internal CUDA NLL forward received invalid tensors"};
  }
  Tensor output{std::move(output_shape), log_probabilities.dtype(), log_probabilities.device()};
  const auto classes{
      static_cast<std::size_t>(log_probabilities.shape()[log_probabilities.rank() - 1])};
  check(spar_cuda_launch_nll_forward(CudaTensorAccess::mutable_data(output),
                                     CudaTensorAccess::data(log_probabilities),
                                     CudaTensorAccess::data(targets), checked_count(output.numel()),
                                     checked_count(classes), checked_count(sequence_length),
                                     dtype_tag(output.dtype()), index_dtype_tag(targets.dtype()),
                                     sequence_length == 0 ? 0 : 1, output.device().index()),
        "cross_entropy NLL forward", output.device());
  return output;
#else
  static_cast<void>(log_probabilities);
  static_cast<void>(targets);
  static_cast<void>(output_shape);
  static_cast<void>(sequence_length);
  unavailable();
#endif
}

Tensor nll_backward(const Tensor& gradient, const Tensor& targets, Shape log_probability_shape,
                    std::size_t sequence_length) {
#if SPAR_ENABLE_CUDA
  if (!gradient.device().is_cuda() || gradient.device() != targets.device() ||
      !gradient.is_contiguous() || !targets.is_contiguous() || log_probability_shape.rank() == 0) {
    throw std::logic_error{"Internal CUDA NLL backward received invalid tensors"};
  }
  Tensor output{zeros(log_probability_shape, gradient.dtype(), gradient.device())};
  const auto classes{
      static_cast<std::size_t>(log_probability_shape[log_probability_shape.rank() - 1])};
  check(spar_cuda_launch_nll_backward(
            CudaTensorAccess::mutable_data(output), CudaTensorAccess::data(gradient),
            CudaTensorAccess::data(targets), checked_count(gradient.numel()),
            checked_count(classes), checked_count(sequence_length), dtype_tag(output.dtype()),
            index_dtype_tag(targets.dtype()), sequence_length == 0 ? 0 : 1,
            output.device().index()),
        "cross_entropy NLL backward", output.device());
  return output;
#else
  static_cast<void>(gradient);
  static_cast<void>(targets);
  static_cast<void>(log_probability_shape);
  static_cast<void>(sequence_length);
  unavailable();
#endif
}

GradientNormSummary gradient_norm_summary(const Tensor& gradient) {
#if SPAR_ENABLE_CUDA
  if (!gradient.device().is_cuda() || !gradient.is_contiguous()) {
    throw std::logic_error{"Internal CUDA gradient norm received an invalid Tensor"};
  }
  SparCudaNormSummary summary{};
  check(spar_cuda_gradient_norm_summary(&summary, CudaTensorAccess::data(gradient),
                                        checked_count(gradient.numel()),
                                        dtype_tag(gradient.dtype()), gradient.device().index()),
        "global gradient norm", gradient.device());
  return {.scale = summary.scale,
          .scaled_sum_squares = summary.scaled_sum_squares,
          .has_infinity = summary.has_infinity != 0,
          .has_nan = summary.has_nan != 0};
#else
  static_cast<void>(gradient);
  unavailable();
#endif
}

void scale_in_place(Tensor& values, double factor) {
#if SPAR_ENABLE_CUDA
  if (!values.device().is_cuda() || !values.is_contiguous()) {
    throw std::logic_error{"Internal CUDA gradient scaling received an invalid Tensor"};
  }
  check(spar_cuda_launch_scale_in_place(CudaTensorAccess::mutable_data(values),
                                        checked_count(values.numel()), dtype_tag(values.dtype()),
                                        factor, values.device().index()),
        "gradient scaling", values.device());
#else
  static_cast<void>(values);
  static_cast<void>(factor);
  unavailable();
#endif
}

void adamw_update(Tensor& parameter, const Tensor& gradient, Tensor& first_moment,
                  Tensor& second_moment, std::uint64_t step, double learning_rate, double beta1,
                  double beta2, double epsilon, double weight_decay) {
#if SPAR_ENABLE_CUDA
  if (!parameter.device().is_cuda() || parameter.device() != gradient.device() ||
      parameter.device() != first_moment.device() || parameter.device() != second_moment.device() ||
      !parameter.is_contiguous() || !gradient.is_contiguous() || !first_moment.is_contiguous() ||
      !second_moment.is_contiguous() || parameter.shape() != gradient.shape() ||
      parameter.shape() != first_moment.shape() || parameter.shape() != second_moment.shape() ||
      parameter.dtype() != gradient.dtype() || parameter.dtype() != first_moment.dtype() ||
      parameter.dtype() != second_moment.dtype() || step == 0) {
    throw std::logic_error{"Internal CUDA AdamW update received invalid tensors"};
  }
  const double first_correction{1.0 - std::pow(beta1, static_cast<double>(step))};
  const double second_correction{1.0 - std::pow(beta2, static_cast<double>(step))};
  check(spar_cuda_launch_adamw(
            CudaTensorAccess::mutable_data(parameter), CudaTensorAccess::data(gradient),
            CudaTensorAccess::mutable_data(first_moment),
            CudaTensorAccess::mutable_data(second_moment), checked_count(parameter.numel()),
            dtype_tag(parameter.dtype()), learning_rate, beta1, beta2, epsilon, weight_decay,
            first_correction, second_correction, parameter.device().index()),
        "AdamW update", parameter.device());
#else
  static_cast<void>(parameter);
  static_cast<void>(gradient);
  static_cast<void>(first_moment);
  static_cast<void>(second_moment);
  static_cast<void>(step);
  static_cast<void>(learning_rate);
  static_cast<void>(beta1);
  static_cast<void>(beta2);
  static_cast<void>(epsilon);
  static_cast<void>(weight_decay);
  unavailable();
#endif
}

} // namespace spar::detail::cuda_ops
