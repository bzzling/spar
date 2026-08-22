module spar.training.gradients;

import std;
import spar.cuda_ops;
import spar.dtype;
import spar.nn.parameter;
import spar.tensor;

using namespace std;

namespace spar::training {
namespace {

[[nodiscard]] vector<nn::Parameter*> unique_active_parameters(span<nn::Parameter> parameters) {
  vector<nn::Parameter*> unique;
  unique.reserve(parameters.size());
  for (nn::Parameter& parameter : parameters) {
    const bool duplicate{ranges::any_of(unique, [&parameter](const nn::Parameter* existing) {
      return existing->shares_identity_with(parameter);
    })};
    if (!duplicate && parameter.requires_grad() && parameter.has_grad()) {
      unique.push_back(&parameter);
    }
  }
  return unique;
}

struct StableSumSquares final {
  double scale{0.0};
  double scaled_sum_squares{1.0};
  bool has_infinity{false};
  bool has_nan{false};

  void add(double value) noexcept {
    const double magnitude{abs(value)};
    if (isnan(magnitude)) {
      has_nan = true;
      return;
    }
    if (isinf(magnitude)) {
      has_infinity = true;
      return;
    }
    if (magnitude == 0.0) {
      return;
    }
    if (scale < magnitude) {
      const double ratio{scale / magnitude};
      scaled_sum_squares = 1.0 + scaled_sum_squares * ratio * ratio;
      scale = magnitude;
    } else {
      const double ratio{magnitude / scale};
      scaled_sum_squares += ratio * ratio;
    }
  }

  void combine(double other_scale, double other_sum_squares, bool other_infinity,
               bool other_nan) noexcept {
    has_nan = has_nan || other_nan;
    has_infinity = has_infinity || other_infinity;
    if (other_scale == 0.0) {
      return;
    }
    if (scale == 0.0) {
      scale = other_scale;
      scaled_sum_squares = other_sum_squares;
      return;
    }
    if (scale < other_scale) {
      const double ratio{scale / other_scale};
      scaled_sum_squares = other_sum_squares + scaled_sum_squares * ratio * ratio;
      scale = other_scale;
    } else {
      const double ratio{other_scale / scale};
      scaled_sum_squares += other_sum_squares * ratio * ratio;
    }
  }

  [[nodiscard]] double norm() const noexcept {
    if (has_nan) {
      return numeric_limits<double>::quiet_NaN();
    }
    if (has_infinity) {
      return numeric_limits<double>::infinity();
    }
    return scale == 0.0 ? 0.0 : scale * sqrt(scaled_sum_squares);
  }
};

template <typename T>
void add_gradient_to_norm(const Tensor& gradient, StableSumSquares& accumulator) {
  for (const T value : gradient.span<T>()) {
    accumulator.add(static_cast<double>(value));
  }
}

template <typename T> void scale_gradient(Tensor& gradient, double factor) {
  for (T& value : gradient.span<T>()) {
    value = static_cast<T>(static_cast<double>(value) * factor);
  }
}

} // namespace

double global_grad_norm(span<nn::Parameter> parameters) {
  StableSumSquares accumulator;
  for (nn::Parameter* parameter : unique_active_parameters(parameters)) {
    const Tensor gradient{parameter->grad()};
    if (gradient.device() != parameter->tensor().device()) {
      throw logic_error{"Active Parameter gradient Device mismatch"};
    }
    if (gradient.dtype() != DType::Float32 && gradient.dtype() != DType::Float64) {
      throw logic_error{"Active Parameter gradient must be floating point"};
    }
    if (gradient.device().is_cuda()) {
      if (!gradient.is_contiguous()) {
        throw logic_error{"Active CUDA Parameter gradient must be contiguous"};
      }
      const auto summary{detail::cuda_ops::gradient_norm_summary(gradient)};
      accumulator.combine(summary.scale, summary.scaled_sum_squares, summary.has_infinity,
                          summary.has_nan);
      continue;
    }
    switch (gradient.dtype()) {
    case DType::Float32:
      add_gradient_to_norm<float>(gradient, accumulator);
      break;
    case DType::Float64:
      add_gradient_to_norm<double>(gradient, accumulator);
      break;
    case DType::Int32:
    case DType::Int64:
      throw logic_error{"Active Parameter gradient must be floating point"};
    }
  }
  return accumulator.norm();
}

void scale_gradients(span<nn::Parameter> parameters, double factor) {
  if (!isfinite(factor)) {
    throw invalid_argument{"Gradient scale factor must be finite"};
  }
  const auto active{unique_active_parameters(parameters)};
  for (const nn::Parameter* parameter : active) {
    const Tensor gradient{parameter->grad()};
    if (gradient.device() != parameter->tensor().device()) {
      throw logic_error{"Active Parameter gradient Device mismatch"};
    }
    if (gradient.dtype() != DType::Float32 && gradient.dtype() != DType::Float64) {
      throw logic_error{"Active Parameter gradient must be floating point"};
    }
    if (gradient.device().is_cuda() && !gradient.is_contiguous()) {
      throw logic_error{"Active CUDA Parameter gradient must be contiguous"};
    }
  }
  for (nn::Parameter* parameter : active) {
    Tensor gradient{parameter->grad()};
    if (gradient.device().is_cuda()) {
      detail::cuda_ops::scale_in_place(gradient, factor);
      continue;
    }
    switch (gradient.dtype()) {
    case DType::Float32:
      scale_gradient<float>(gradient, factor);
      break;
    case DType::Float64:
      scale_gradient<double>(gradient, factor);
      break;
    case DType::Int32:
    case DType::Int64:
      throw logic_error{"Active Parameter gradient must be floating point"};
    }
  }
}

ClipGradNormResult clip_grad_norm(span<nn::Parameter> parameters, double max_norm) {
  if (!isfinite(max_norm) || max_norm < 0.0) {
    throw invalid_argument{"Maximum gradient norm must be finite and nonnegative"};
  }
  const double total_norm{global_grad_norm(parameters)};
  if (!isfinite(total_norm)) {
    throw runtime_error{"Cannot clip nonfinite gradients"};
  }
  if (total_norm == 0.0 || total_norm <= max_norm) {
    return {.total_norm = total_norm, .scale = 1.0, .clipped = false};
  }
  const double scale{max_norm / total_norm};
  scale_gradients(parameters, scale);
  return {.total_norm = total_norm, .scale = scale, .clipped = true};
}

} // namespace spar::training
