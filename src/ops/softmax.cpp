module spar.ops.softmax;

import std;
import spar.dtype;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_softmax_input(const Tensor& input) {
  if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
    throw invalid_argument{"softmax currently supports floating-point dtypes only"};
  }
  if (input.numel() == 0) {
    throw invalid_argument{"softmax is undefined for an empty tensor"};
  }
  if (!input.is_contiguous()) {
    throw invalid_argument{"softmax currently requires a contiguous tensor"};
  }
  if (input.requires_grad()) {
    throw logic_error{"autograd for softmax is not implemented yet"};
  }
}

template <typename T> Tensor softmax_values(const Tensor& input) {
  const auto input_values{input.span<T>()};
  const T positive_infinity{numeric_limits<T>::infinity()};
  const T not_a_number{numeric_limits<T>::quiet_NaN()};

  size_t positive_infinity_count{0};
  for (const T value : input_values) {
    if (isnan(value)) {
      Tensor output{input.shape(), input.dtype()};
      output.fill<T>(not_a_number);
      return output;
    }
    if (value == positive_infinity) {
      ++positive_infinity_count;
    }
  }

  Tensor output{input.shape(), input.dtype()};
  auto output_values{output.span<T>()};

  if (positive_infinity_count != 0) {
    const T probability{T{1} / static_cast<T>(positive_infinity_count)};
    for (size_t index{0}; index < output_values.size(); ++index) {
      output_values[index] = input_values[index] == positive_infinity ? probability : T{0};
    }
    return output;
  }

  T maximum{input_values[0]};
  for (size_t index{1}; index < input_values.size(); ++index) {
    if (input_values[index] > maximum) {
      maximum = input_values[index];
    }
  }

  if (maximum == -positive_infinity) {
    output.fill<T>(not_a_number);
    return output;
  }

  T exponential_sum{0};
  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = std::exp(input_values[index] - maximum);
    exponential_sum += output_values[index];
  }
  for (T& value : output_values) {
    value /= exponential_sum;
  }
  return output;
}

} // namespace

Tensor softmax(const Tensor& input) {
  validate_softmax_input(input);
  switch (input.dtype()) {
  case DType::Float32:
    return softmax_values<float>(input);
  case DType::Float64:
    return softmax_values<double>(input);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"softmax dtype validation invariant violated"};
  }
  throw logic_error{"softmax dtype validation invariant violated"};
}

} // namespace spar
