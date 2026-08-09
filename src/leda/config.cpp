module spar.leda.config;

import std;
import spar.dtype;
import spar.nn.decoder;
import spar.shape;

using namespace std;

namespace spar::leda {
namespace {

void validate(const LedaConfig& config) {
  if (config.vocab_size == 0 || config.model_dim == 0 || config.hidden_dim == 0 ||
      config.num_layers == 0 || config.num_query_heads == 0 || config.num_kv_heads == 0) {
    throw invalid_argument{"Leda dimensions, layer count, and head counts must be positive"};
  }
  if (config.dtype != DType::Float32 && config.dtype != DType::Float64) {
    throw invalid_argument{"Leda requires a floating-point dtype"};
  }
  if (!isfinite(config.norm_epsilon) || config.norm_epsilon <= 0.0 ||
      !isfinite(config.qk_norm_epsilon) || config.qk_norm_epsilon <= 0.0) {
    throw invalid_argument{"Leda normalization epsilons must be finite and positive"};
  }
  if (!isfinite(config.rope_theta) || config.rope_theta <= 0.0) {
    throw invalid_argument{"Leda RoPE theta must be finite and positive"};
  }
  if (config.model_dim % config.num_query_heads != 0) {
    throw invalid_argument{"Leda model dimension must be divisible by query heads"};
  }
  if (config.num_query_heads % config.num_kv_heads != 0) {
    throw invalid_argument{"Leda query heads must be divisible by KV heads"};
  }
  if ((config.model_dim / config.num_query_heads) % 2 != 0) {
    throw invalid_argument{"Leda head dimension must be even for RoPE"};
  }
  const auto maximum{static_cast<size_t>(numeric_limits<Shape::dimension_type>::max())};
  if (config.vocab_size > maximum || config.model_dim > maximum || config.hidden_dim > maximum) {
    throw overflow_error{"Leda dimension exceeds Spar Shape representation"};
  }
}

} // namespace

nn::DecoderConfig decoder_config(const LedaConfig& config) {
  validate(config);
  return {.vocab_size = config.vocab_size,
          .model_dim = config.model_dim,
          .hidden_dim = config.hidden_dim,
          .num_layers = config.num_layers,
          .num_query_heads = config.num_query_heads,
          .num_kv_heads = config.num_kv_heads,
          .dtype = config.dtype,
          .attention_bias = config.attention_bias,
          .mlp_bias = config.mlp_bias,
          .norm_epsilon = config.norm_epsilon,
          .rope_theta = config.rope_theta,
          .qk_norm = config.qk_norm,
          .qk_norm_epsilon = config.qk_norm_epsilon};
}

LedaConfig leda_tiny(size_t vocab_size, DType dtype) {
  LedaConfig config{.vocab_size = vocab_size,
                    .model_dim = 128,
                    .hidden_dim = 384,
                    .num_layers = 4,
                    .num_query_heads = 4,
                    .num_kv_heads = 2,
                    .dtype = dtype,
                    .qk_norm = true,
                    .norm_epsilon = 1.0e-5,
                    .qk_norm_epsilon = 1.0e-6,
                    .rope_theta = 10000.0,
                    .attention_bias = false,
                    .mlp_bias = false};
  static_cast<void>(decoder_config(config));
  return config;
}

LedaConfig leda_small(size_t vocab_size, DType dtype) {
  LedaConfig config{.vocab_size = vocab_size,
                    .model_dim = 256,
                    .hidden_dim = 768,
                    .num_layers = 8,
                    .num_query_heads = 8,
                    .num_kv_heads = 2,
                    .dtype = dtype,
                    .qk_norm = true,
                    .norm_epsilon = 1.0e-5,
                    .qk_norm_epsilon = 1.0e-6,
                    .rope_theta = 10000.0,
                    .attention_bias = false,
                    .mlp_bias = false};
  static_cast<void>(decoder_config(config));
  return config;
}

} // namespace spar::leda
