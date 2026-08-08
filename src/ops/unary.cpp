module spar.ops.unary;

import std;
import spar.dtype;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_unary_input(const Tensor& input) {
  if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
    throw invalid_argument{"Unary operations currently support floating-point dtypes only"};
  }
  if (!input.is_contiguous()) {
    throw invalid_argument{"Unary operations currently require contiguous tensors"};
  }
}

template <typename T, typename Operation>
Tensor apply_unary(const Tensor& input, Operation operation) {
  Tensor output{input.shape(), input.dtype()};
  const auto input_values{input.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = operation(input_values[index]);
  }
  return output;
}

template <typename Operation>
Tensor dispatch_unary(const Tensor& input, string_view operation_name, Operation operation) {
  validate_unary_input(input);
  if (input.requires_grad()) {
    throw logic_error{"autograd for " + string{operation_name} + " is not implemented yet"};
  }
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
  return dispatch_unary(input, "negate", [](auto value) { return -value; });
}

Tensor square(const Tensor& input) {
  return dispatch_unary(input, "square", [](auto value) { return value * value; });
}

Tensor reciprocal(const Tensor& input) {
  return dispatch_unary(input, "reciprocal", [](auto value) { return decltype(value){1} / value; });
}

Tensor exp(const Tensor& input) {
  return dispatch_unary(input, "exp", [](auto value) { return std::exp(value); });
}

Tensor log(const Tensor& input) {
  return dispatch_unary(input, "log", [](auto value) { return std::log(value); });
}

Tensor sqrt(const Tensor& input) {
  return dispatch_unary(input, "sqrt", [](auto value) { return std::sqrt(value); });
}

Tensor sigmoid(const Tensor& input) {
  return dispatch_unary(input, "sigmoid", [](auto value) { return stable_sigmoid(value); });
}

Tensor silu(const Tensor& input) {
  return dispatch_unary(input, "silu", [](auto value) { return value * stable_sigmoid(value); });
}

} // namespace spar
