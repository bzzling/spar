module spar.optim.adamw;

import std;
import spar.dtype;
import spar.nn.parameter;
import spar.tensor;

using namespace std;

namespace spar::optim {
namespace {

void validate_learning_rate(double learning_rate) {
  if (!isfinite(learning_rate) || learning_rate < 0.0) {
    throw invalid_argument{"AdamW learning_rate must be finite and nonnegative"};
  }
}

void validate_hyperparameters(double learning_rate, double beta1, double beta2, double epsilon,
                              double weight_decay) {
  validate_learning_rate(learning_rate);
  if (!isfinite(beta1) || beta1 < 0.0 || beta1 >= 1.0) {
    throw invalid_argument{"AdamW beta1 must be finite and in [0,1)"};
  }
  if (!isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0) {
    throw invalid_argument{"AdamW beta2 must be finite and in [0,1)"};
  }
  if (!isfinite(epsilon) || epsilon <= 0.0) {
    throw invalid_argument{"AdamW epsilon must be finite and positive"};
  }
  if (!isfinite(weight_decay) || weight_decay < 0.0) {
    throw invalid_argument{"AdamW weight_decay must be finite and nonnegative"};
  }
}

template <typename T>
void update_values(Tensor& values, const Tensor& gradient, Tensor& first_moment,
                   Tensor& second_moment, uint64_t step, double learning_rate, double beta1,
                   double beta2, double epsilon, double weight_decay) {
  spar::detail::validate_same_device(values, gradient, "AdamW update");
  spar::detail::validate_same_device(values, first_moment, "AdamW update");
  spar::detail::validate_same_device(values, second_moment, "AdamW update");
  auto parameter_values{values.span<T>()};
  const auto gradient_values{gradient.span<T>()};
  auto first_values{first_moment.span<T>()};
  auto second_values{second_moment.span<T>()};
  const double first_correction{1.0 - pow(beta1, static_cast<double>(step))};
  const double second_correction{1.0 - pow(beta2, static_cast<double>(step))};
  const double decay_factor{1.0 - learning_rate * weight_decay};
  for (size_t index{0}; index < parameter_values.size(); ++index) {
    const double gradient_value{static_cast<double>(gradient_values[index])};
    first_values[index] = static_cast<T>(beta1 * static_cast<double>(first_values[index]) +
                                         (1.0 - beta1) * gradient_value);
    second_values[index] = static_cast<T>(beta2 * static_cast<double>(second_values[index]) +
                                          (1.0 - beta2) * gradient_value * gradient_value);
    const double corrected_first{static_cast<double>(first_values[index]) / first_correction};
    const double corrected_second{static_cast<double>(second_values[index]) / second_correction};
    const double decayed{static_cast<double>(parameter_values[index]) * decay_factor};
    parameter_values[index] = static_cast<T>(decayed - learning_rate * corrected_first /
                                                           (sqrt(corrected_second) + epsilon));
  }
}

} // namespace

AdamW::AdamW(vector<nn::Parameter> parameters, double learning_rate, double beta1, double beta2,
             double epsilon, double weight_decay)
    : learning_rate_{learning_rate}, beta1_{beta1}, beta2_{beta2}, epsilon_{epsilon},
      weight_decay_{weight_decay} {
  validate_hyperparameters(learning_rate, beta1, beta2, epsilon, weight_decay);
  entries_.reserve(parameters.size());
  for (auto& parameter : parameters) {
    const bool duplicate{ranges::any_of(entries_, [&parameter](const Entry& entry) {
      return entry.parameter.shares_identity_with(parameter);
    })};
    if (!duplicate) {
      entries_.push_back(Entry{std::move(parameter), nullopt});
    }
  }
}

void AdamW::step() {
  for (const Entry& entry : entries_) {
    const nn::Parameter& parameter{entry.parameter};
    if (!parameter.requires_grad() || !parameter.has_grad()) {
      continue;
    }
    const Tensor gradient{parameter.grad()};
    if (gradient.shape() != parameter.tensor().shape() ||
        gradient.dtype() != parameter.tensor().dtype() ||
        gradient.device() != parameter.tensor().device()) {
      throw logic_error{"AdamW gradient shape, dtype, or Device does not match its Parameter"};
    }
    detail::require_cpu(parameter.tensor(), "AdamW step");
  }

  for (Entry& entry : entries_) {
    nn::Parameter& parameter{entry.parameter};
    if (!parameter.requires_grad() || !parameter.has_grad()) {
      continue;
    }
    const Tensor gradient{parameter.grad()};
    if (!entry.state) {
      entry.state.emplace(zeros(parameter.tensor().shape(), parameter.tensor().dtype(),
                                parameter.tensor().device()),
                          zeros(parameter.tensor().shape(), parameter.tensor().dtype(),
                                parameter.tensor().device()),
                          0);
    }
    State& state{*entry.state};
    if (state.step == numeric_limits<uint64_t>::max()) {
      throw overflow_error{"AdamW per-Parameter step counter overflow"};
    }
    ++state.step;
    Tensor values{parameter.tensor().detach()};
    switch (parameter.tensor().dtype()) {
    case DType::Float32:
      update_values<float>(values, gradient, state.first_moment, state.second_moment, state.step,
                           learning_rate_, beta1_, beta2_, epsilon_, weight_decay_);
      break;
    case DType::Float64:
      update_values<double>(values, gradient, state.first_moment, state.second_moment, state.step,
                            learning_rate_, beta1_, beta2_, epsilon_, weight_decay_);
      break;
    case DType::Int32:
    case DType::Int64:
      throw logic_error{"AdamW encountered a non-floating Parameter"};
    }
  }
}

void AdamW::zero_grad() {
  for (Entry& entry : entries_) {
    entry.parameter.zero_grad();
  }
}

size_t AdamW::parameter_count() const noexcept {
  return entries_.size();
}
double AdamW::learning_rate() const noexcept {
  return learning_rate_;
}
void AdamW::set_learning_rate(double learning_rate) {
  validate_learning_rate(learning_rate);
  learning_rate_ = learning_rate;
}
double AdamW::beta1() const noexcept {
  return beta1_;
}
double AdamW::beta2() const noexcept {
  return beta2_;
}
double AdamW::epsilon() const noexcept {
  return epsilon_;
}
double AdamW::weight_decay() const noexcept {
  return weight_decay_;
}

bool AdamW::tracks(const nn::Parameter& parameter) const noexcept {
  return ranges::any_of(entries_, [&parameter](const Entry& entry) {
    return entry.parameter.shares_identity_with(parameter);
  });
}

optional<AdamWParameterState> AdamW::parameter_state(const nn::Parameter& parameter) const {
  const auto iterator{ranges::find_if(entries_, [&parameter](const Entry& entry) {
    return entry.parameter.shares_identity_with(parameter);
  })};
  if (iterator == entries_.end()) {
    throw invalid_argument{"AdamW does not track the requested Parameter"};
  }
  if (!iterator->state) {
    return nullopt;
  }
  return AdamWParameterState{iterator->state->first_moment.detach().clone(),
                             iterator->state->second_moment.detach().clone(),
                             iterator->state->step};
}

void AdamW::set_parameter_state(const nn::Parameter& parameter,
                                optional<AdamWParameterState> state) {
  const auto iterator{ranges::find_if(entries_, [&parameter](const Entry& entry) {
    return entry.parameter.shares_identity_with(parameter);
  })};
  if (iterator == entries_.end()) {
    throw invalid_argument{"AdamW does not track the requested Parameter"};
  }
  if (!state) {
    iterator->state.reset();
    return;
  }
  if (state->step == 0) {
    throw invalid_argument{"AdamW restored state step must be at least one"};
  }
  for (const Tensor* moment : {&state->first_moment, &state->second_moment}) {
    if (moment->shape() != parameter.tensor().shape() ||
        moment->dtype() != parameter.tensor().dtype() ||
        moment->device() != parameter.tensor().device()) {
      throw invalid_argument{"AdamW restored moment shape, dtype, or Device mismatch"};
    }
    if (moment->dtype() != DType::Float32 && moment->dtype() != DType::Float64) {
      throw invalid_argument{"AdamW restored moments must be floating point"};
    }
  }
  iterator->state.emplace(state->first_moment.detach().clone(),
                          state->second_moment.detach().clone(), state->step);
}

} // namespace spar::optim
