module spar.ops.reduction;

import std;
import spar.dtype;
import spar.ops.scalar;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_reduction_input(const Tensor& input) {
  if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
    throw invalid_argument{"Reductions currently support floating-point dtypes only"};
  }
}

template <typename T> Tensor sum_values(const Tensor& input) {
  T result{0};
  for (size_t index{0}; index < input.numel(); ++index) {
    result += detail::logical_value<T>(input, index);
  }

  Tensor output{Shape{}, input.dtype()};
  output.span<T>()[0] = result;
  return output;
}

template <typename T> Tensor mean_values(const Tensor& input) {
  T result{0};
  for (size_t index{0}; index < input.numel(); ++index) {
    result += detail::logical_value<T>(input, index);
  }
  result /= static_cast<T>(input.numel());

  Tensor output{Shape{}, input.dtype()};
  output.span<T>()[0] = result;
  return output;
}

template <typename T> Tensor max_values(const Tensor& input) {
  T result{detail::logical_value<T>(input, 0)};
  for (size_t index{1}; index < input.numel(); ++index) {
    const T value{detail::logical_value<T>(input, index)};
    if (isnan(value)) {
      result = value;
      break;
    }
    if (value > result) {
      result = value;
    }
  }

  Tensor output{Shape{}, input.dtype()};
  output.span<T>()[0] = result;
  return output;
}

vector<size_t> validate_axes(const Tensor& input, span<const size_t> axes) {
  if (axes.empty()) {
    throw invalid_argument{"Reduction axes must not be empty"};
  }
  vector<size_t> normalized{axes.begin(), axes.end()};
  ranges::sort(normalized);
  for (size_t index{0}; index < normalized.size(); ++index) {
    if (normalized[index] >= input.rank()) {
      throw out_of_range{"Reduction axis is out of range"};
    }
    if (index != 0 && normalized[index] == normalized[index - 1]) {
      throw invalid_argument{"Reduction axes must not contain duplicates"};
    }
  }
  return normalized;
}

Shape reduced_shape(const Tensor& input, span<const size_t> axes, bool keepdim) {
  vector<bool> reduced(input.rank(), false);
  for (const size_t axis : axes) {
    reduced[axis] = true;
  }
  vector<Shape::dimension_type> dimensions;
  dimensions.reserve(keepdim ? input.rank() : input.rank() - axes.size());
  for (size_t axis{0}; axis < input.rank(); ++axis) {
    if (reduced[axis]) {
      if (keepdim) {
        dimensions.push_back(1);
      }
    } else {
      dimensions.push_back(input.shape()[axis]);
    }
  }
  return Shape{std::move(dimensions)};
}

size_t reduced_element_count(const Tensor& input, span<const size_t> axes) {
  size_t count{1};
  for (const size_t axis : axes) {
    const auto extent{static_cast<size_t>(input.shape()[axis])};
    if (extent != 0 && count > numeric_limits<size_t>::max() / extent) {
      throw overflow_error{"Reduction element count overflow"};
    }
    count *= extent;
  }
  return count;
}

template <typename T>
Tensor axis_sum_values(const Tensor& input, span<const size_t> axes, bool keepdim,
                       const Shape& output_shape) {
  Tensor output{zeros(output_shape, input.dtype())};
  auto output_values{output.span<T>()};
  const auto output_strides{output_shape.contiguous_strides()};
  vector<bool> reduced(input.rank(), false);
  for (const size_t axis : axes) {
    reduced[axis] = true;
  }
  vector<size_t> coordinates(input.rank(), 0);

  for (size_t logical_index{0}; logical_index < input.numel(); ++logical_index) {
    size_t remaining{logical_index};
    for (size_t index{input.rank()}; index > 0; --index) {
      const size_t axis{index - 1};
      const auto extent{static_cast<size_t>(input.shape()[axis])};
      coordinates[axis] = remaining % extent;
      remaining /= extent;
    }

    size_t output_index{0};
    size_t output_axis{0};
    for (size_t axis{0}; axis < input.rank(); ++axis) {
      if (keepdim) {
        if (!reduced[axis]) {
          output_index += coordinates[axis] * output_strides[axis];
        }
      } else if (!reduced[axis]) {
        output_index += coordinates[axis] * output_strides[output_axis++];
      }
    }
    output_values[output_index] += detail::logical_value<T>(input, logical_index);
  }
  return output;
}

Tensor restore_and_expand_gradient(const Tensor& gradient, const Shape& input_shape,
                                   span<const size_t> axes, bool keepdim) {
  Tensor restored{gradient};
  if (!keepdim) {
    vector<bool> reduced(input_shape.rank(), false);
    for (const size_t axis : axes) {
      reduced[axis] = true;
    }
    vector<Shape::dimension_type> dimensions(input_shape.rank(), 1);
    size_t gradient_axis{0};
    for (size_t axis{0}; axis < input_shape.rank(); ++axis) {
      if (!reduced[axis]) {
        dimensions[axis] = gradient.shape()[gradient_axis++];
      }
    }
    restored = gradient.reshape(Shape{std::move(dimensions)});
  }
  return restored.expand(input_shape).contiguous();
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

Tensor sum(const Tensor& input, span<const size_t> axes, bool keepdim) {
  validate_reduction_input(input);
  vector<size_t> normalized_axes{validate_axes(input, axes)};
  const Shape output_shape{reduced_shape(input, normalized_axes, keepdim)};
  Tensor output{input.dtype() == DType::Float32
                    ? axis_sum_values<float>(input, normalized_axes, keepdim, output_shape)
                    : axis_sum_values<double>(input, normalized_axes, keepdim, output_shape)};
  if (input.requires_grad()) {
    const Shape input_shape{input.shape()};
    detail::record_operation(output, {input},
                             [input_shape, normalized_axes = std::move(normalized_axes),
                              keepdim](const Tensor& gradient) {
                               return vector<Tensor>{restore_and_expand_gradient(
                                   gradient, input_shape, normalized_axes, keepdim)};
                             });
  }
  return output;
}

Tensor sum(const Tensor& input, size_t axis, bool keepdim) {
  return sum(input, span<const size_t>{&axis, 1}, keepdim);
}

Tensor sum(const Tensor& input, initializer_list<size_t> axes, bool keepdim) {
  return sum(input, span<const size_t>{axes.begin(), axes.size()}, keepdim);
}

Tensor mean(const Tensor& input) {
  validate_reduction_input(input);
  if (input.numel() == 0) {
    throw invalid_argument{"mean is undefined for an empty tensor"};
  }
  Tensor output{input.dtype() == DType::Float32 ? mean_values<float>(input)
                                                : mean_values<double>(input)};
  if (input.requires_grad()) {
    const Shape input_shape{input.shape()};
    const DType input_dtype{input.dtype()};
    const size_t input_numel{input.numel()};
    detail::record_operation(
        output, {input}, [input_shape, input_dtype, input_numel](const Tensor& gradient) {
          Tensor contribution{input_shape, input_dtype};
          if (input_dtype == DType::Float32) {
            contribution.fill<float>(gradient.span<float>()[0] / static_cast<float>(input_numel));
          } else {
            contribution.fill<double>(gradient.span<double>()[0] /
                                      static_cast<double>(input_numel));
          }
          return vector<Tensor>{std::move(contribution)};
        });
  }
  return output;
}

Tensor mean(const Tensor& input, span<const size_t> axes, bool keepdim) {
  validate_reduction_input(input);
  vector<size_t> normalized_axes{validate_axes(input, axes)};
  const size_t element_count{reduced_element_count(input, normalized_axes)};
  if (element_count == 0) {
    throw invalid_argument{"mean is undefined for an empty reduction domain"};
  }
  const Shape output_shape{reduced_shape(input, normalized_axes, keepdim)};
  Tensor summed{input.dtype() == DType::Float32
                    ? axis_sum_values<float>(input, normalized_axes, keepdim, output_shape)
                    : axis_sum_values<double>(input, normalized_axes, keepdim, output_shape)};
  Tensor output{divide_scalar(summed, static_cast<double>(element_count))};
  if (input.requires_grad()) {
    const Shape input_shape{input.shape()};
    detail::record_operation(
        output, {input},
        [input_shape, normalized_axes = std::move(normalized_axes), keepdim,
         element_count](const Tensor& gradient) {
          const auto expanded{
              restore_and_expand_gradient(gradient, input_shape, normalized_axes, keepdim)};
          return vector<Tensor>{divide_scalar(expanded, static_cast<double>(element_count))};
        });
  }
  return output;
}

Tensor mean(const Tensor& input, size_t axis, bool keepdim) {
  return mean(input, span<const size_t>{&axis, 1}, keepdim);
}

Tensor mean(const Tensor& input, initializer_list<size_t> axes, bool keepdim) {
  return mean(input, span<const size_t>{axes.begin(), axes.size()}, keepdim);
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
