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
  return spar::detail::shares_autograd_identity(tensor_, other.tensor_);
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

void move_to(Parameter& parameter, Device target) {
  move_to(span<Parameter>{&parameter, 1}, target);
}

void move_to(span<Parameter> parameters, Device target) {
  vector<Parameter> unique;
  unique.reserve(parameters.size());
  for (Parameter& parameter : parameters) {
    const bool duplicate{ranges::any_of(unique, [&parameter](const Parameter& existing) {
      return existing.shares_identity_with(parameter);
    })};
    if (!duplicate) {
      unique.push_back(parameter);
    }
  }

  for (const Parameter& parameter : unique) {
    if (parameter.tensor_.device() != target && parameter.has_grad()) {
      throw logic_error{"Parameter Device migration requires clearing its live gradient first"};
    }
  }

  struct StagedMigration final {
    Parameter parameter;
    Tensor value;
  };
  vector<StagedMigration> staged;
  staged.reserve(unique.size());
  for (Parameter& parameter : unique) {
    if (parameter.tensor_.device() != target) {
      staged.push_back(StagedMigration{parameter, parameter.tensor_.detach().to(target)});
    }
  }

  for (StagedMigration& migration : staged) {
    spar::detail::swap_storage_payloads(migration.parameter.tensor_, migration.value);
  }
}

} // namespace spar::nn
