module spar.nn.mlp;

import std;
import spar.dtype;
import spar.nn.functional;
import spar.nn.linear;
import spar.random;
import spar.tensor;

using namespace std;

namespace spar::nn {
namespace {

size_t validated_model_dim(size_t model_dim, size_t hidden_dim, DType dtype) {
  if (model_dim == 0 || hidden_dim == 0) {
    throw invalid_argument{"SwiGLUMLP dimensions must be positive"};
  }
  if (dtype != DType::Float32 && dtype != DType::Float64) {
    throw invalid_argument{"SwiGLUMLP requires a floating-point dtype"};
  }
  return model_dim;
}

} // namespace

SwiGLUMLP::SwiGLUMLP(size_t model_dim, size_t hidden_dim, DType dtype, Random& random, bool bias)
    : model_dim_{validated_model_dim(model_dim, hidden_dim, dtype)}, hidden_dim_{hidden_dim},
      gate_proj_{model_dim_, hidden_dim, dtype, random, bias},
      up_proj_{model_dim, hidden_dim, dtype, random, bias},
      down_proj_{hidden_dim, model_dim, dtype, random, bias} {}

Tensor SwiGLUMLP::forward(const Tensor& input) const {
  if (input.rank() < 2) {
    throw invalid_argument{"SwiGLUMLP input must have rank at least 2"};
  }
  if (static_cast<size_t>(input.shape()[input.rank() - 1]) != model_dim_) {
    throw invalid_argument{"SwiGLUMLP input final dimension does not match model_dim"};
  }
  if (input.dtype() != gate_proj_.weight().tensor().dtype()) {
    throw invalid_argument{"SwiGLUMLP input dtype does not match its projections"};
  }
  const Tensor gate{gate_proj_.forward(input)};
  const Tensor up{up_proj_.forward(input)};
  return down_proj_.forward(swiglu(gate, up));
}

size_t SwiGLUMLP::model_dim() const noexcept {
  return model_dim_;
}
size_t SwiGLUMLP::hidden_dim() const noexcept {
  return hidden_dim_;
}
Linear& SwiGLUMLP::gate_proj() noexcept {
  return gate_proj_;
}
const Linear& SwiGLUMLP::gate_proj() const noexcept {
  return gate_proj_;
}
Linear& SwiGLUMLP::up_proj() noexcept {
  return up_proj_;
}
const Linear& SwiGLUMLP::up_proj() const noexcept {
  return up_proj_;
}
Linear& SwiGLUMLP::down_proj() noexcept {
  return down_proj_;
}
const Linear& SwiGLUMLP::down_proj() const noexcept {
  return down_proj_;
}

} // namespace spar::nn
