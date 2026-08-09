export module spar.nn.decoder;

import std;
export import spar.dtype;
export import spar.nn.embedding;
export import spar.nn.rmsnorm;
export import spar.nn.transformer;
export import spar.random;
export import spar.tensor;

export namespace spar::nn {

struct DecoderConfig final {
  std::size_t vocab_size{};
  std::size_t model_dim{};
  std::size_t hidden_dim{};
  std::size_t num_layers{};
  std::size_t num_query_heads{};
  std::size_t num_kv_heads{};

  DType dtype{DType::Float32};
  bool attention_bias{false};
  bool mlp_bias{false};
  double norm_epsilon{1.0e-5};
  double rope_theta{10000.0};
  bool qk_norm{false};
  double qk_norm_epsilon{1.0e-6};
};

class DecoderLM {
public:
  DecoderLM(DecoderConfig config, Random& random);

  DecoderLM(const DecoderLM&) = delete;
  DecoderLM& operator=(const DecoderLM&) = delete;
  DecoderLM(DecoderLM&&) noexcept = default;
  DecoderLM& operator=(DecoderLM&&) noexcept = default;

  [[nodiscard]] Tensor forward(const Tensor& token_ids, std::size_t start_position = 0) const;

  [[nodiscard]] const DecoderConfig& config() const noexcept;
  [[nodiscard]] std::size_t vocab_size() const noexcept;
  [[nodiscard]] std::size_t model_dim() const noexcept;
  [[nodiscard]] std::size_t num_layers() const noexcept;

  [[nodiscard]] Embedding& token_embedding() noexcept;
  [[nodiscard]] const Embedding& token_embedding() const noexcept;
  [[nodiscard]] TransformerBlock& block(std::size_t index);
  [[nodiscard]] const TransformerBlock& block(std::size_t index) const;
  [[nodiscard]] std::span<TransformerBlock> blocks() noexcept;
  [[nodiscard]] std::span<const TransformerBlock> blocks() const noexcept;
  [[nodiscard]] RMSNorm& final_norm() noexcept;
  [[nodiscard]] const RMSNorm& final_norm() const noexcept;

private:
  struct ValidatedConfiguration final {
    DecoderConfig value;
  };

  DecoderLM(ValidatedConfiguration configuration, Random& random);
  [[nodiscard]] static ValidatedConfiguration validate_configuration(DecoderConfig config);
  [[nodiscard]] static std::vector<TransformerBlock> make_blocks(const DecoderConfig& config,
                                                                 Random& random);

  DecoderConfig config_;
  Embedding token_embedding_;
  std::vector<TransformerBlock> blocks_;
  RMSNorm final_norm_;
};

} // namespace spar::nn
