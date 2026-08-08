module spar.ops.scalar;

import std;
import spar.dtype;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_scalar_input(const Tensor& input) {
  if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
    throw invalid_argument{"Scalar operations currently support floating-point dtypes only"};
  }
  if (!input.is_contiguous()) {
    throw invalid_argument{"Scalar operations currently require contiguous tensors"};
  }
}

template <typename T, typename Operation>
Tensor apply_scalar(const Tensor& input, T scalar, Operation operation) {
  Tensor output{input.shape(), input.dtype()};
  const auto input_values{input.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = operation(input_values[index], scalar);
  }
  return output;
}

template <typename Operation>
Tensor dispatch_scalar(const Tensor& input, double value, Operation operation) {
  validate_scalar_input(input);
  switch (input.dtype()) {
  case DType::Float32: {
    constexpr double float_limit{static_cast<double>(numeric_limits<float>::max())};
    if (isfinite(value) && (value > float_limit || value < -float_limit)) {
      throw overflow_error{"Scalar value is outside the finite Float32 range"};
    }
    return apply_scalar<float>(input, static_cast<float>(value), operation);
  }
  case DType::Float64:
    return apply_scalar<double>(input, value, operation);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Scalar dtype validation invariant violated"};
  }
  throw logic_error{"Scalar dtype validation invariant violated"};
}

} // namespace

Tensor add_scalar(const Tensor& input, double value) {
  return dispatch_scalar(input, value, [](auto element, auto scalar) { return element + scalar; });
}

Tensor subtract_scalar(const Tensor& input, double value) {
  return dispatch_scalar(input, value, [](auto element, auto scalar) { return element - scalar; });
}

Tensor multiply_scalar(const Tensor& input, double value) {
  return dispatch_scalar(input, value, [](auto element, auto scalar) { return element * scalar; });
}

Tensor divide_scalar(const Tensor& input, double value) {
  return dispatch_scalar(input, value, [](auto element, auto scalar) { return element / scalar; });
}

} // namespace spar
