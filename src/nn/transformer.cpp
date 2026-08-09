module spar.nn.transformer;

import std;
import spar.dtype;
import spar.nn.attention;
import spar.nn.mlp;
import spar.nn.rmsnorm;
import spar.ops.elementwise;
import spar.random;
import spar.tensor;

using namespace std;

namespace spar::nn {

TransformerBlock::Configuration TransformerBlock::validate_configuration(
    size_t model_dim, size_t hidden_dim, size_t num_query_heads, size_t num_kv_heads,
    double norm_epsilon, double rope_theta, bool qk_norm, double qk_norm_epsilon) {
  if (model_dim == 0 || hidden_dim == 0) {
    throw invalid_argument{"TransformerBlock dimensions must be positive"};
  }
  if (!isfinite(norm_epsilon) || norm_epsilon <= 0.0) {
    throw invalid_argument{"TransformerBlock norm_epsilon must be finite and positive"};
  }
  return Configuration{model_dim,    hidden_dim, num_query_heads, num_kv_heads,
                       norm_epsilon, rope_theta, qk_norm,         qk_norm_epsilon};
}

TransformerBlock::TransformerBlock(size_t model_dim, size_t hidden_dim, size_t num_query_heads,
                                   size_t num_kv_heads, DType dtype, Random& random,
                                   bool attention_bias, bool mlp_bias, double norm_epsilon,
                                   double rope_theta, bool qk_norm, double qk_norm_epsilon)
    : TransformerBlock{validate_configuration(model_dim, hidden_dim, num_query_heads, num_kv_heads,
                                              norm_epsilon, rope_theta, qk_norm, qk_norm_epsilon),
                       dtype, random, attention_bias, mlp_bias} {}

TransformerBlock::TransformerBlock(Configuration configuration, DType dtype, Random& random,
                                   bool attention_bias, bool mlp_bias)
    : model_dim_{configuration.model_dim}, hidden_dim_{configuration.hidden_dim},
      attention_norm_{model_dim_, dtype, configuration.norm_epsilon},
      attention_{model_dim_,
                 configuration.num_query_heads,
                 configuration.num_kv_heads,
                 dtype,
                 random,
                 attention_bias,
                 configuration.rope_theta,
                 configuration.qk_norm,
                 configuration.qk_norm_epsilon},
      mlp_norm_{model_dim_, dtype, configuration.norm_epsilon},
      mlp_{model_dim_, hidden_dim_, dtype, random, mlp_bias} {}

Tensor TransformerBlock::forward(const Tensor& input, size_t start_position) const {
  if (input.rank() != 3) {
    throw invalid_argument{"TransformerBlock input must have shape [B,T,model_dim]"};
  }
  if (input.shape()[0] <= 0 || input.shape()[1] <= 0) {
    throw invalid_argument{"TransformerBlock does not support empty batches or sequences"};
  }
  if (static_cast<size_t>(input.shape()[2]) != model_dim_) {
    throw invalid_argument{"TransformerBlock input final dimension does not match model_dim"};
  }
  if (input.dtype() != attention_norm_.weight().tensor().dtype()) {
    throw invalid_argument{"TransformerBlock input dtype does not match its layers"};
  }
  const Tensor hidden{
      add(input, attention_.forward(attention_norm_.forward(input), start_position))};
  return add(hidden, mlp_.forward(mlp_norm_.forward(hidden)));
}

size_t TransformerBlock::model_dim() const noexcept {
  return model_dim_;
}
size_t TransformerBlock::hidden_dim() const noexcept {
  return hidden_dim_;
}
RMSNorm& TransformerBlock::attention_norm() noexcept {
  return attention_norm_;
}
const RMSNorm& TransformerBlock::attention_norm() const noexcept {
  return attention_norm_;
}
SelfAttention& TransformerBlock::attention() noexcept {
  return attention_;
}
const SelfAttention& TransformerBlock::attention() const noexcept {
  return attention_;
}
RMSNorm& TransformerBlock::mlp_norm() noexcept {
  return mlp_norm_;
}
const RMSNorm& TransformerBlock::mlp_norm() const noexcept {
  return mlp_norm_;
}
SwiGLUMLP& TransformerBlock::mlp() noexcept {
  return mlp_;
}
const SwiGLUMLP& TransformerBlock::mlp() const noexcept {
  return mlp_;
}

} // namespace spar::nn
