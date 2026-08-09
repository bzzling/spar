export module spar.nn.attention;

import std;
export import spar.dtype;
export import spar.nn.linear;
export import spar.random;
export import spar.tensor;

export namespace spar::nn {

[[nodiscard]] Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key,
                                                  const Tensor& value, bool causal = false);

class SelfAttention {
public:
  SelfAttention(std::size_t model_dim, std::size_t num_query_heads, std::size_t num_kv_heads,
                DType dtype, Random& random, bool bias = false, double rope_theta = 10000.0);

  [[nodiscard]] Tensor forward(const Tensor& input, std::size_t start_position = 0) const;

  [[nodiscard]] std::size_t model_dim() const noexcept;
  [[nodiscard]] std::size_t num_query_heads() const noexcept;
  [[nodiscard]] std::size_t num_kv_heads() const noexcept;
  [[nodiscard]] std::size_t head_dim() const noexcept;
  [[nodiscard]] double rope_theta() const noexcept;

  [[nodiscard]] Linear& q_proj() noexcept;
  [[nodiscard]] const Linear& q_proj() const noexcept;
  [[nodiscard]] Linear& k_proj() noexcept;
  [[nodiscard]] const Linear& k_proj() const noexcept;
  [[nodiscard]] Linear& v_proj() noexcept;
  [[nodiscard]] const Linear& v_proj() const noexcept;
  [[nodiscard]] Linear& out_proj() noexcept;
  [[nodiscard]] const Linear& out_proj() const noexcept;

private:
  struct Configuration final {
    std::size_t model_dim;
    std::size_t num_query_heads;
    std::size_t num_kv_heads;
    std::size_t head_dim;
    double rope_theta;
  };

  SelfAttention(Configuration configuration, DType dtype, Random& random, bool bias);
  [[nodiscard]] static Configuration validate_configuration(std::size_t model_dim,
                                                            std::size_t num_query_heads,
                                                            std::size_t num_kv_heads, DType dtype,
                                                            double rope_theta);

  std::size_t model_dim_;
  std::size_t num_query_heads_;
  std::size_t num_kv_heads_;
  std::size_t head_dim_;
  double rope_theta_;
  Linear q_proj_;
  Linear k_proj_;
  Linear v_proj_;
  Linear out_proj_;
};

} // namespace spar::nn
