module spar.nn.rmsnorm;

import std;
import spar.dtype;
import spar.nn.parameter;
import spar.ops.elementwise;
import spar.ops.reduction;
import spar.ops.scalar;
import spar.ops.unary;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar::nn {
namespace {

Tensor make_rmsnorm_weight(size_t normalized_size, DType dtype, double epsilon) {
  if (normalized_size == 0) {
    throw invalid_argument{"RMSNorm normalized_size must be positive"};
  }
  if (!isfinite(epsilon) || epsilon <= 0.0) {
    throw invalid_argument{"RMSNorm epsilon must be finite and positive"};
  }
  if (dtype != DType::Float32 && dtype != DType::Float64) {
    throw invalid_argument{"RMSNorm requires a floating-point dtype"};
  }
  if (normalized_size > static_cast<size_t>(numeric_limits<Shape::dimension_type>::max())) {
    throw overflow_error{"RMSNorm normalized_size exceeds the Shape representation"};
  }
  return ones(Shape{static_cast<Shape::dimension_type>(normalized_size)}, dtype);
}

} // namespace

RMSNorm::RMSNorm(size_t normalized_size, DType dtype, double epsilon)
    : normalized_size_{normalized_size}, epsilon_{epsilon},
      weight_{make_rmsnorm_weight(normalized_size, dtype, epsilon)} {}

Tensor RMSNorm::forward(const Tensor& input) const {
  if (input.rank() == 0) {
    throw invalid_argument{"RMSNorm input must have rank at least 1"};
  }
  if (static_cast<size_t>(input.shape()[input.rank() - 1]) != normalized_size_) {
    throw invalid_argument{"RMSNorm input final dimension does not match normalized_size"};
  }
  if (input.dtype() != weight_.tensor().dtype()) {
    throw invalid_argument{"RMSNorm input dtype does not match its weight"};
  }
  const Tensor mean_square{mean(square(input), input.rank() - 1, true)};
  const Tensor denominator{sqrt(add_scalar(mean_square, epsilon_))};
  return multiply(divide(input, denominator), weight_.tensor());
}

size_t RMSNorm::normalized_size() const noexcept {
  return normalized_size_;
}

double RMSNorm::epsilon() const noexcept {
  return epsilon_;
}

Parameter& RMSNorm::weight() noexcept {
  return weight_;
}

const Parameter& RMSNorm::weight() const noexcept {
  return weight_;
}

} // namespace spar::nn
