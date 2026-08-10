module spar.ops.unary;

import std;
import spar.dtype;
import spar.ops.elementwise;
import spar.ops.scalar;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_unary_input(const Tensor& input) {
  detail::require_cpu(input, "Unary operation");
  if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
    throw invalid_argument{"Unary operations currently support floating-point dtypes only"};
  }
  if (!input.is_contiguous()) {
    throw invalid_argument{"Unary operations currently require contiguous tensors"};
  }
}

template <typename T, typename Operation>
Tensor apply_unary(const Tensor& input, Operation operation) {
  Tensor output{input.shape(), input.dtype(), input.device()};
  const auto input_values{input.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = operation(input_values[index]);
  }
  return output;
}

template <typename Operation> Tensor dispatch_unary(const Tensor& input, Operation operation) {
  validate_unary_input(input);
  switch (input.dtype()) {
  case DType::Float32:
    return apply_unary<float>(input, operation);
  case DType::Float64:
    return apply_unary<double>(input, operation);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Unary dtype validation invariant violated"};
  }
  throw logic_error{"Unary dtype validation invariant violated"};
}

template <typename T> T stable_sigmoid(T value) {
  if (value >= T{0}) {
    return T{1} / (T{1} + std::exp(-value));
  }
  const T exponential{std::exp(value)};
  return exponential / (T{1} + exponential);
}

} // namespace

Tensor negate(const Tensor& input) {
  Tensor output{dispatch_unary(input, [](auto value) { return -value; })};
  if (input.requires_grad()) {
    detail::record_operation(output, {input}, [](const Tensor& gradient) {
      return vector<Tensor>{multiply_scalar(gradient, -1.0)};
    });
  }
  return output;
}

Tensor square(const Tensor& input) {
  Tensor output{dispatch_unary(input, [](auto value) { return value * value; })};
  if (input.requires_grad()) {
    const Tensor saved_input{input.detach().clone()};
    detail::record_operation(output, {input}, [saved_input](const Tensor& gradient) {
      return vector<Tensor>{multiply(gradient, multiply_scalar(saved_input, 2.0))};
    });
  }
  return output;
}

Tensor reciprocal(const Tensor& input) {
  Tensor output{dispatch_unary(input, [](auto value) { return decltype(value){1} / value; })};
  if (input.requires_grad()) {
    const Tensor saved_input{input.detach().clone()};
    detail::record_operation(output, {input}, [saved_input](const Tensor& gradient) {
      const auto denominator{multiply(saved_input, saved_input)};
      return vector<Tensor>{multiply_scalar(divide(gradient, denominator), -1.0)};
    });
  }
  return output;
}

Tensor exp(const Tensor& input) {
  Tensor output{dispatch_unary(input, [](auto value) { return std::exp(value); })};
  if (input.requires_grad()) {
    const Tensor saved_output{output.detach().clone()};
    detail::record_operation(output, {input}, [saved_output](const Tensor& gradient) {
      return vector<Tensor>{multiply(gradient, saved_output)};
    });
  }
  return output;
}

Tensor log(const Tensor& input) {
  Tensor output{dispatch_unary(input, [](auto value) { return std::log(value); })};
  if (input.requires_grad()) {
    const Tensor saved_input{input.detach().clone()};
    detail::record_operation(output, {input}, [saved_input](const Tensor& gradient) {
      return vector<Tensor>{divide(gradient, saved_input)};
    });
  }
  return output;
}

Tensor sqrt(const Tensor& input) {
  Tensor output{dispatch_unary(input, [](auto value) { return std::sqrt(value); })};
  if (input.requires_grad()) {
    const Tensor saved_output{output.detach().clone()};
    detail::record_operation(output, {input}, [saved_output](const Tensor& gradient) {
      return vector<Tensor>{divide(gradient, multiply_scalar(saved_output, 2.0))};
    });
  }
  return output;
}

Tensor sigmoid(const Tensor& input) {
  Tensor output{dispatch_unary(input, [](auto value) { return stable_sigmoid(value); })};
  if (input.requires_grad()) {
    const Tensor saved_output{output.detach().clone()};
    detail::record_operation(output, {input}, [saved_output](const Tensor& gradient) {
      const auto one_minus_output{add_scalar(multiply_scalar(saved_output, -1.0), 1.0)};
      return vector<Tensor>{multiply(gradient, multiply(saved_output, one_minus_output))};
    });
  }
  return output;
}

Tensor silu(const Tensor& input) {
  Tensor output{dispatch_unary(input, [](auto value) { return value * stable_sigmoid(value); })};
  if (input.requires_grad()) {
    const Tensor saved_input{input.detach().clone()};
    detail::record_operation(output, {input}, [saved_input](const Tensor& gradient) {
      const auto sigmoid_value{sigmoid(saved_input)};
      const auto one_minus_sigmoid{add_scalar(multiply_scalar(sigmoid_value, -1.0), 1.0)};
      const auto sigmoid_slope{multiply(sigmoid_value, one_minus_sigmoid)};
      const auto local_derivative{add(sigmoid_value, multiply(saved_input, sigmoid_slope))};
      return vector<Tensor>{multiply(gradient, local_derivative)};
    });
  }
  return output;
}

} // namespace spar
