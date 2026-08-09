module spar.nn.linear;

import std;
import spar.dtype;
import spar.nn.parameter;
import spar.ops.elementwise;
import spar.ops.matmul;
import spar.random;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar::nn {
namespace {

void validate_linear_configuration(size_t in_features, size_t out_features, DType dtype) {
  if (in_features == 0 || out_features == 0) {
    throw invalid_argument{"Linear feature dimensions must be positive"};
  }
  if (dtype != DType::Float32 && dtype != DType::Float64) {
    throw invalid_argument{"Linear requires a floating-point dtype"};
  }
  const auto maximum_dimension{static_cast<size_t>(numeric_limits<Shape::dimension_type>::max())};
  if (in_features > maximum_dimension || out_features > maximum_dimension) {
    throw overflow_error{"Linear feature dimension exceeds the Shape representation"};
  }
}

Tensor make_linear_weight(size_t in_features, size_t out_features, DType dtype, Random& random) {
  validate_linear_configuration(in_features, out_features, dtype);
  const double bound{1.0 / sqrt(static_cast<double>(in_features))};
  return random_uniform(Shape{static_cast<Shape::dimension_type>(in_features),
                              static_cast<Shape::dimension_type>(out_features)},
                        dtype, random, -bound, bound);
}

} // namespace

Linear::Linear(size_t in_features, size_t out_features, DType dtype, Random& random, bool bias)
    : in_features_{in_features}, out_features_{out_features},
      weight_{make_linear_weight(in_features, out_features, dtype, random)} {
  if (bias) {
    bias_.emplace(zeros(Shape{static_cast<Shape::dimension_type>(out_features)}, dtype));
  }
}

Tensor Linear::forward(const Tensor& input) const {
  if (input.rank() < 2) {
    throw invalid_argument{"Linear input must have rank at least 2"};
  }
  if (static_cast<size_t>(input.shape()[input.rank() - 1]) != in_features_) {
    throw invalid_argument{"Linear input final dimension does not match in_features"};
  }
  if (input.dtype() != weight_.tensor().dtype()) {
    throw invalid_argument{"Linear input dtype does not match its weight"};
  }
  Tensor output{matmul(input, weight_.tensor())};
  if (bias_) {
    output = add(output, bias_->tensor());
  }
  return output;
}

size_t Linear::in_features() const noexcept {
  return in_features_;
}

size_t Linear::out_features() const noexcept {
  return out_features_;
}

bool Linear::has_bias() const noexcept {
  return bias_.has_value();
}

Parameter& Linear::weight() noexcept {
  return weight_;
}

const Parameter& Linear::weight() const noexcept {
  return weight_;
}

Parameter* Linear::bias() noexcept {
  return bias_ ? &*bias_ : nullptr;
}

const Parameter* Linear::bias() const noexcept {
  return bias_ ? &*bias_ : nullptr;
}

} // namespace spar::nn
