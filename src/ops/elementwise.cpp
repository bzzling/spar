module spar.ops.elementwise;

import std;
import spar.dtype;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_binary_inputs(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape()) {
    throw invalid_argument{"Elementwise operations require identical shapes"};
  }
  if (a.dtype() != b.dtype()) {
    throw invalid_argument{"Elementwise operations require identical dtypes"};
  }
  if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64) {
    throw invalid_argument{"Elementwise operations currently support floating-point dtypes only"};
  }
  if (!a.is_contiguous() || !b.is_contiguous()) {
    throw invalid_argument{"Elementwise operations currently require contiguous tensors"};
  }
}

template <typename T> Tensor add_values(const Tensor& a, const Tensor& b) {
  Tensor output{a.shape(), a.dtype()};
  const auto a_values{a.span<T>()};
  const auto b_values{b.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = a_values[index] + b_values[index];
  }
  return output;
}

template <typename T> Tensor subtract_values(const Tensor& a, const Tensor& b) {
  Tensor output{a.shape(), a.dtype()};
  const auto a_values{a.span<T>()};
  const auto b_values{b.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = a_values[index] - b_values[index];
  }
  return output;
}

template <typename T> Tensor multiply_values(const Tensor& a, const Tensor& b) {
  Tensor output{a.shape(), a.dtype()};
  const auto a_values{a.span<T>()};
  const auto b_values{b.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = a_values[index] * b_values[index];
  }
  return output;
}

template <typename T> Tensor divide_values(const Tensor& a, const Tensor& b) {
  Tensor output{a.shape(), a.dtype()};
  const auto a_values{a.span<T>()};
  const auto b_values{b.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = a_values[index] / b_values[index];
  }
  return output;
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b) {
  validate_binary_inputs(a, b);
  switch (a.dtype()) {
  case DType::Float32:
    return add_values<float>(a, b);
  case DType::Float64:
    return add_values<double>(a, b);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Elementwise dtype validation invariant violated"};
  }
  throw logic_error{"Elementwise dtype validation invariant violated"};
}

Tensor subtract(const Tensor& a, const Tensor& b) {
  validate_binary_inputs(a, b);
  switch (a.dtype()) {
  case DType::Float32:
    return subtract_values<float>(a, b);
  case DType::Float64:
    return subtract_values<double>(a, b);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Elementwise dtype validation invariant violated"};
  }
  throw logic_error{"Elementwise dtype validation invariant violated"};
}

Tensor multiply(const Tensor& a, const Tensor& b) {
  validate_binary_inputs(a, b);
  switch (a.dtype()) {
  case DType::Float32:
    return multiply_values<float>(a, b);
  case DType::Float64:
    return multiply_values<double>(a, b);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Elementwise dtype validation invariant violated"};
  }
  throw logic_error{"Elementwise dtype validation invariant violated"};
}

Tensor divide(const Tensor& a, const Tensor& b) {
  validate_binary_inputs(a, b);
  switch (a.dtype()) {
  case DType::Float32:
    return divide_values<float>(a, b);
  case DType::Float64:
    return divide_values<double>(a, b);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Elementwise dtype validation invariant violated"};
  }
  throw logic_error{"Elementwise dtype validation invariant violated"};
}

} // namespace spar
