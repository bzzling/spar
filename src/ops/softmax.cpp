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
}

void validate_axis_softmax_input(const Tensor& input, size_t axis) {
  if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
    throw invalid_argument{"softmax currently supports floating-point dtypes only"};
  }
  if (axis >= input.rank()) {
    throw out_of_range{"softmax axis is out of range"};
  }
  if (input.numel() == 0) {
    throw invalid_argument{"softmax is undefined for an empty tensor"};
  }
}

struct AxisSoftmaxResult final {
  Tensor output;
  vector<unsigned char> undefined_slices;
};

template <typename T> AxisSoftmaxResult axis_softmax_values(const Tensor& input, size_t axis) {
  const auto axis_extent{static_cast<size_t>(input.shape()[axis])};
  size_t outer{1};
  for (size_t index{0}; index < axis; ++index) {
    outer *= static_cast<size_t>(input.shape()[index]);
  }
  size_t inner{1};
  for (size_t index{axis + 1}; index < input.rank(); ++index) {
    inner *= static_cast<size_t>(input.shape()[index]);
  }

  Tensor output{input.shape(), input.dtype()};
  auto output_values{output.span<T>()};
  vector<unsigned char> undefined_slices(outer * inner, 0);
  const T infinity{numeric_limits<T>::infinity()};
  const T nan{numeric_limits<T>::quiet_NaN()};

  for (size_t outer_index{0}; outer_index < outer; ++outer_index) {
    for (size_t inner_index{0}; inner_index < inner; ++inner_index) {
      const size_t slice_index{outer_index * inner + inner_index};
      size_t positive_infinity_count{0};
      bool contains_nan{false};
      T maximum{-infinity};
      for (size_t axis_index{0}; axis_index < axis_extent; ++axis_index) {
        const size_t logical_index{(outer_index * axis_extent + axis_index) * inner + inner_index};
        const T value{detail::logical_value<T>(input, logical_index)};
        contains_nan = contains_nan || isnan(value);
        positive_infinity_count += value == infinity ? 1 : 0;
        if (value > maximum) {
          maximum = value;
        }
      }

      if (contains_nan || maximum == -infinity) {
        undefined_slices[slice_index] = 1;
        for (size_t axis_index{0}; axis_index < axis_extent; ++axis_index) {
          output_values[(outer_index * axis_extent + axis_index) * inner + inner_index] = nan;
        }
        continue;
      }
      if (positive_infinity_count != 0) {
        undefined_slices[slice_index] = 1;
        const T probability{T{1} / static_cast<T>(positive_infinity_count)};
        for (size_t axis_index{0}; axis_index < axis_extent; ++axis_index) {
          const size_t logical_index{(outer_index * axis_extent + axis_index) * inner +
                                     inner_index};
          output_values[logical_index] =
              detail::logical_value<T>(input, logical_index) == infinity ? probability : T{0};
        }
        continue;
      }

      T exponential_sum{0};
      for (size_t axis_index{0}; axis_index < axis_extent; ++axis_index) {
        const size_t logical_index{(outer_index * axis_extent + axis_index) * inner + inner_index};
        output_values[logical_index] =
            std::exp(detail::logical_value<T>(input, logical_index) - maximum);
        exponential_sum += output_values[logical_index];
      }
      for (size_t axis_index{0}; axis_index < axis_extent; ++axis_index) {
        output_values[(outer_index * axis_extent + axis_index) * inner + inner_index] /=
            exponential_sum;
      }
    }
  }
  return AxisSoftmaxResult{std::move(output), std::move(undefined_slices)};
}

template <typename T>
Tensor axis_softmax_gradient(const Tensor& gradient, const Tensor& saved_output, size_t axis,
                             span<const unsigned char> undefined_slices) {
  const auto axis_extent{static_cast<size_t>(saved_output.shape()[axis])};
  size_t outer{1};
  for (size_t index{0}; index < axis; ++index) {
    outer *= static_cast<size_t>(saved_output.shape()[index]);
  }
  size_t inner{1};
  for (size_t index{axis + 1}; index < saved_output.rank(); ++index) {
    inner *= static_cast<size_t>(saved_output.shape()[index]);
  }

  const auto gradient_values{gradient.span<T>()};
  const auto output_values{saved_output.span<T>()};
  Tensor contribution{saved_output.shape(), saved_output.dtype()};
  auto contribution_values{contribution.span<T>()};
  const T nan{numeric_limits<T>::quiet_NaN()};
  for (size_t outer_index{0}; outer_index < outer; ++outer_index) {
    for (size_t inner_index{0}; inner_index < inner; ++inner_index) {
      const size_t slice_index{outer_index * inner + inner_index};
      if (undefined_slices[slice_index] != 0) {
        for (size_t axis_index{0}; axis_index < axis_extent; ++axis_index) {
          contribution_values[(outer_index * axis_extent + axis_index) * inner + inner_index] = nan;
        }
        continue;
      }
      T dot{0};
      for (size_t axis_index{0}; axis_index < axis_extent; ++axis_index) {
        const size_t index{(outer_index * axis_extent + axis_index) * inner + inner_index};
        dot += gradient_values[index] * output_values[index];
      }
      for (size_t axis_index{0}; axis_index < axis_extent; ++axis_index) {
        const size_t index{(outer_index * axis_extent + axis_index) * inner + inner_index};
        contribution_values[index] = output_values[index] * (gradient_values[index] - dot);
      }
    }
  }
  return contribution;
}

template <typename T> bool has_undefined_softmax_derivative(const Tensor& input) {
  const auto values{input.span<T>()};
  const T positive_infinity{numeric_limits<T>::infinity()};
  bool all_negative_infinity{true};
  for (const T value : values) {
    if (isnan(value) || value == positive_infinity) {
      return true;
    }
    if (value != -positive_infinity) {
      all_negative_infinity = false;
    }
  }
  return all_negative_infinity;
}

template <typename T>
Tensor softmax_gradient(const Tensor& gradient, const Tensor& saved_output,
                        bool undefined_derivative) {
  Tensor contribution{saved_output.shape(), saved_output.dtype()};
  if (undefined_derivative) {
    contribution.fill<T>(numeric_limits<T>::quiet_NaN());
    return contribution;
  }

  const auto gradient_values{gradient.span<T>()};
  const auto output_values{saved_output.span<T>()};
  auto contribution_values{contribution.span<T>()};
  T dot{0};
  for (size_t index{0}; index < output_values.size(); ++index) {
    dot += gradient_values[index] * output_values[index];
  }
  for (size_t index{0}; index < output_values.size(); ++index) {
    contribution_values[index] = output_values[index] * (gradient_values[index] - dot);
  }
  return contribution;
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
  Tensor output{input.dtype() == DType::Float32 ? softmax_values<float>(input)
                                                : softmax_values<double>(input)};
  if (input.requires_grad()) {
    const bool undefined_derivative{input.dtype() == DType::Float32
                                        ? has_undefined_softmax_derivative<float>(input)
                                        : has_undefined_softmax_derivative<double>(input)};
    const Tensor saved_output{output.detach().clone()};
    const DType dtype{input.dtype()};
    detail::record_operation(
        output, {input}, [saved_output, undefined_derivative, dtype](const Tensor& gradient) {
          if (dtype == DType::Float32) {
            return vector<Tensor>{
                softmax_gradient<float>(gradient, saved_output, undefined_derivative)};
          }
          return vector<Tensor>{
              softmax_gradient<double>(gradient, saved_output, undefined_derivative)};
        });
  }
  return output;
}

Tensor softmax(const Tensor& input, size_t axis) {
  validate_axis_softmax_input(input, axis);
  AxisSoftmaxResult result{input.dtype() == DType::Float32
                               ? axis_softmax_values<float>(input, axis)
                               : axis_softmax_values<double>(input, axis)};
  Tensor output{std::move(result.output)};
  if (input.requires_grad()) {
    const Tensor saved_output{output.detach().clone()};
    const DType dtype{input.dtype()};
    detail::record_operation(
        output, {input},
        [saved_output, axis, undefined_slices = std::move(result.undefined_slices),
         dtype](const Tensor& gradient) {
          if (dtype == DType::Float32) {
            return vector<Tensor>{
                axis_softmax_gradient<float>(gradient, saved_output, axis, undefined_slices)};
          }
          return vector<Tensor>{
              axis_softmax_gradient<double>(gradient, saved_output, axis, undefined_slices)};
        });
  }
  return output;
}

} // namespace spar
