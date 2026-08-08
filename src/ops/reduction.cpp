module spar.ops.reduction;

import std;
import spar.dtype;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_reduction_input(const Tensor& input) {
  if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
    throw invalid_argument{"Reductions currently support floating-point dtypes only"};
  }
  if (!input.is_contiguous()) {
    throw invalid_argument{"Reductions currently require contiguous tensors"};
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
  validate_reduction_input(input);
  Tensor output{input.dtype() == DType::Float32 ? sum_values<float>(input)
                                                : sum_values<double>(input)};
  if (input.requires_grad()) {
    const Shape input_shape{input.shape()};
    const DType input_dtype{input.dtype()};
    detail::record_operation(output, {input}, [input_shape, input_dtype](const Tensor& gradient) {
      Tensor contribution{input_shape, input_dtype};
      if (input_dtype == DType::Float32) {
        contribution.fill<float>(gradient.span<float>()[0]);
      } else {
        contribution.fill<double>(gradient.span<double>()[0]);
      }
      return vector<Tensor>{std::move(contribution)};
    });
  }
  return output;
}

Tensor mean(const Tensor& input) {
  validate_reduction_input(input);
  if (input.requires_grad()) {
    throw logic_error{"autograd for mean is not implemented yet"};
  }
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
  validate_reduction_input(input);
  if (input.requires_grad()) {
    throw logic_error{"autograd for reduce_max is not implemented yet"};
  }
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
