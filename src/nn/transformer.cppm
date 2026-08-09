export module spar.nn.transformer;

import std;
export import spar.dtype;
export import spar.nn.attention;
export import spar.nn.mlp;
export import spar.nn.rmsnorm;
export import spar.random;
export import spar.tensor;

export namespace spar::nn {

class TransformerBlock {
public:
  TransformerBlock(std::size_t model_dim, std::size_t hidden_dim, std::size_t num_query_heads,
                   std::size_t num_kv_heads, DType dtype, Random& random,
                   bool attention_bias = false, bool mlp_bias = false, double norm_epsilon = 1.0e-5,
                   double rope_theta = 10000.0, bool qk_norm = false,
                   double qk_norm_epsilon = 1.0e-6);

  [[nodiscard]] Tensor forward(const Tensor& input, std::size_t start_position = 0) const;

  [[nodiscard]] std::size_t model_dim() const noexcept;
  [[nodiscard]] std::size_t hidden_dim() const noexcept;

  [[nodiscard]] RMSNorm& attention_norm() noexcept;
  [[nodiscard]] const RMSNorm& attention_norm() const noexcept;
  [[nodiscard]] SelfAttention& attention() noexcept;
  [[nodiscard]] const SelfAttention& attention() const noexcept;
  [[nodiscard]] RMSNorm& mlp_norm() noexcept;
  [[nodiscard]] const RMSNorm& mlp_norm() const noexcept;
  [[nodiscard]] SwiGLUMLP& mlp() noexcept;
  [[nodiscard]] const SwiGLUMLP& mlp() const noexcept;

private:
  struct Configuration final {
    std::size_t model_dim;
    std::size_t hidden_dim;
    std::size_t num_query_heads;
    std::size_t num_kv_heads;
    double norm_epsilon;
    double rope_theta;
    bool qk_norm;
    double qk_norm_epsilon;
  };

  TransformerBlock(Configuration configuration, DType dtype, Random& random, bool attention_bias,
                   bool mlp_bias);
  [[nodiscard]] static Configuration
  validate_configuration(std::size_t model_dim, std::size_t hidden_dim, std::size_t num_query_heads,
                         std::size_t num_kv_heads, double norm_epsilon, double rope_theta,
                         bool qk_norm, double qk_norm_epsilon);

  std::size_t model_dim_;
  std::size_t hidden_dim_;
  RMSNorm attention_norm_;
  SelfAttention attention_;
  RMSNorm mlp_norm_;
  SwiGLUMLP mlp_;
};

} // namespace spar::nn
