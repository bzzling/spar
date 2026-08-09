export module spar.leda.config;

import std;
export import spar.dtype;
export import spar.nn.decoder;

export namespace spar::leda {

/// Leda v0 is a dense causal decoder with pre-norm RMSNorm blocks, GQA, optional QK-Norm, RoPE,
/// SwiGLU, residual connections, and a tied token-embedding output projection.
struct LedaConfig final {
  std::size_t vocab_size;
  std::size_t model_dim;
  std::size_t hidden_dim;
  std::size_t num_layers;
  std::size_t num_query_heads;
  std::size_t num_kv_heads;
  DType dtype{DType::Float32};
  bool qk_norm{true};
  double norm_epsilon{1.0e-5};
  double qk_norm_epsilon{1.0e-6};
  double rope_theta{10000.0};
  bool attention_bias{false};
  bool mlp_bias{false};
};

/// Explicitly validates and maps the Leda family configuration to Spar's generic decoder.
[[nodiscard]] nn::DecoderConfig decoder_config(const LedaConfig& config);

/// CPU reference-development presets, not production-scale Leda configurations.
[[nodiscard]] LedaConfig leda_tiny(std::size_t vocab_size, DType dtype = DType::Float32);
[[nodiscard]] LedaConfig leda_small(std::size_t vocab_size, DType dtype = DType::Float32);

} // namespace spar::leda
