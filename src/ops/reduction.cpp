module spar.ops.reduction;

import std;
import spar.dtype;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_reduction_dtype(DType dtype) {
  if (dtype != DType::Float32 && dtype != DType::Float64) {
    throw invalid_argument{"Reductions currently support floating-point dtypes only"};
  }
}

template <typename T> Tensor sum_values(const Tensor& input) {
  T result{0};
  for (const T value : input.span<T>()) {
    result += value;
  }

  Tensor output{Shape{}, input.dtype()};
  output.span<T>()[0] = result;
  return output;
}

template <typename T> Tensor mean_values(const Tensor& input) {
  T result{0};
  for (const T value : input.span<T>()) {
    result += value;
  }
  result /= static_cast<T>(input.numel());

  Tensor output{Shape{}, input.dtype()};
  output.span<T>()[0] = result;
  return output;
}

template <typename T> Tensor max_values(const Tensor& input) {
  const auto values{input.span<T>()};
  T result{values[0]};
  for (size_t index{1}; index < values.size(); ++index) {
    if (isnan(values[index])) {
      result = values[index];
      break;
    }
    if (values[index] > result) {
      result = values[index];
    }
  }

  Tensor output{Shape{}, input.dtype()};
  output.span<T>()[0] = result;
  return output;
}

} // namespace

Tensor sum(const Tensor& input) {
  validate_reduction_dtype(input.dtype());
  switch (input.dtype()) {
  case DType::Float32:
    return sum_values<float>(input);
  case DType::Float64:
    return sum_values<double>(input);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Reduction dtype validation invariant violated"};
  }
  throw logic_error{"Reduction dtype validation invariant violated"};
}

Tensor mean(const Tensor& input) {
  validate_reduction_dtype(input.dtype());
  if (input.numel() == 0) {
    throw invalid_argument{"mean is undefined for an empty tensor"};
  }
  switch (input.dtype()) {
  case DType::Float32:
    return mean_values<float>(input);
  case DType::Float64:
    return mean_values<double>(input);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Reduction dtype validation invariant violated"};
  }
  throw logic_error{"Reduction dtype validation invariant violated"};
}

Tensor reduce_max(const Tensor& input) {
  validate_reduction_dtype(input.dtype());
  if (input.numel() == 0) {
    throw invalid_argument{"reduce_max is undefined for an empty tensor"};
  }
  switch (input.dtype()) {
  case DType::Float32:
    return max_values<float>(input);
  case DType::Float64:
    return max_values<double>(input);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Reduction dtype validation invariant violated"};
  }
  throw logic_error{"Reduction dtype validation invariant violated"};
}

} // namespace spar
