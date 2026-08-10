module spar.nn.rope;

import std;
import spar.dtype;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar::nn {
namespace {

void validate_rope(const Tensor& input, size_t start_position, double theta) {
  if (input.rank() < 2) {
    throw invalid_argument{"apply_rope input must have rank at least 2"};
  }
  if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
    throw invalid_argument{"apply_rope requires a floating-point dtype"};
  }
  const auto feature_count{input.shape()[input.rank() - 1]};
  if (feature_count <= 0 || feature_count % 2 != 0) {
    throw invalid_argument{"apply_rope final dimension must be positive and even"};
  }
  if (!isfinite(theta) || theta <= 0.0) {
    throw invalid_argument{"apply_rope theta must be finite and positive"};
  }
  const size_t sequence_length{static_cast<size_t>(input.shape()[input.rank() - 2])};
  if (sequence_length != 0 &&
      start_position > numeric_limits<size_t>::max() - sequence_length + 1) {
    throw overflow_error{"apply_rope position range overflow"};
  }
}

template <typename T>
Tensor rotate(const Tensor& input, size_t start_position, double theta, bool inverse) {
  Tensor output{input.shape(), input.dtype(), input.device()};
  if (input.numel() == 0) {
    return output;
  }
  const size_t sequence_length{static_cast<size_t>(input.shape()[input.rank() - 2])};
  const size_t feature_count{static_cast<size_t>(input.shape()[input.rank() - 1])};
  const size_t outer_count{input.numel() / (sequence_length * feature_count)};
  auto output_values{output.span<T>()};
  for (size_t outer{0}; outer < outer_count; ++outer) {
    for (size_t token{0}; token < sequence_length; ++token) {
      const double position{static_cast<double>(start_position + token)};
      for (size_t pair{0}; pair < feature_count / 2; ++pair) {
        const double exponent{-2.0 * static_cast<double>(pair) /
                              static_cast<double>(feature_count)};
        const double angle{position * pow(theta, exponent)};
        const T cosine{static_cast<T>(cos(angle))};
        T sine{static_cast<T>(sin(angle))};
        if (inverse) {
          sine = -sine;
        }
        const size_t index{(outer * sequence_length + token) * feature_count + 2 * pair};
        const T first{detail::logical_value<T>(input, index)};
        const T second{detail::logical_value<T>(input, index + 1)};
        output_values[index] = first * cosine - second * sine;
        output_values[index + 1] = first * sine + second * cosine;
      }
    }
  }
  return output;
}

} // namespace

Tensor apply_rope(const Tensor& input, size_t start_position, double theta) {
  validate_rope(input, start_position, theta);
  Tensor output{input.dtype() == DType::Float32
                    ? rotate<float>(input, start_position, theta, false)
                    : rotate<double>(input, start_position, theta, false)};
  if (input.requires_grad()) {
    const DType dtype{input.dtype()};
    detail::record_operation(
        output, {input}, [start_position, theta, dtype](const Tensor& gradient) {
          if (dtype == DType::Float32) {
            return vector<Tensor>{rotate<float>(gradient, start_position, theta, true)};
          }
          return vector<Tensor>{rotate<double>(gradient, start_position, theta, true)};
        });
  }
  return output;
}

} // namespace spar::nn
