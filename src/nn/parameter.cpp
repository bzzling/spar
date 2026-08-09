module spar.nn.parameter;

import std;
import spar.dtype;
import spar.tensor;

using namespace std;

namespace spar::nn {

Parameter::Parameter(Tensor initial_value) : tensor_{initial_value.detach().clone()} {
  if (tensor_.dtype() != DType::Float32 && tensor_.dtype() != DType::Float64) {
    throw invalid_argument{"Parameter requires a floating-point Tensor"};
  }
  tensor_.set_requires_grad();
}

const Tensor& Parameter::tensor() const noexcept {
  return tensor_;
}

bool Parameter::requires_grad() const noexcept {
  return tensor_.requires_grad();
}

bool Parameter::shares_identity_with(const Parameter& other) const noexcept {
  return detail::shares_autograd_identity(tensor_, other.tensor_);
}

void Parameter::set_requires_grad(bool enabled) {
  tensor_.set_requires_grad(enabled);
}

bool Parameter::has_grad() const noexcept {
  return tensor_.has_grad();
}

Tensor Parameter::grad() const {
  return tensor_.grad();
}

void Parameter::zero_grad() {
  tensor_.zero_grad();
}

Parameter Parameter::clone() const {
  return Parameter{tensor_};
}

} // namespace spar::nn
