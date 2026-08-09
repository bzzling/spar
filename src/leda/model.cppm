export module spar.leda.model;

import std;
export import spar.leda.config;
export import spar.nn.parameters;
export import spar.random;
export import spar.tensor;

export namespace spar::leda {

class Leda final {
public:
  Leda(LedaConfig config, Random& random);

  Leda(const Leda&) = delete;
  Leda& operator=(const Leda&) = delete;
  Leda(Leda&&) noexcept = default;
  Leda& operator=(Leda&&) noexcept = default;

  /// Adopts a checkpoint-loaded DecoderLM after verifying exact mapped configuration equality.
  [[nodiscard]] static Leda from_decoder(LedaConfig config, nn::DecoderLM decoder);

  [[nodiscard]] Tensor forward(const Tensor& token_ids, std::size_t start_position = 0) const;
  [[nodiscard]] const LedaConfig& config() const noexcept;
  [[nodiscard]] nn::DecoderLM& decoder() noexcept;
  [[nodiscard]] const nn::DecoderLM& decoder() const noexcept;

private:
  Leda(LedaConfig config, nn::DecoderLM decoder, int);

  LedaConfig config_;
  nn::DecoderLM decoder_;
};

[[nodiscard]] std::vector<nn::NamedParameter> named_parameters(Leda& model);
[[nodiscard]] std::vector<nn::Parameter> parameters(Leda& model);
void zero_grad(Leda& model);

struct ModelStatistics final {
  std::uint64_t total_parameters;
  std::uint64_t trainable_parameters;
  std::uint64_t parameter_bytes;
};

[[nodiscard]] ModelStatistics model_statistics(Leda& model);

struct AdamWMemoryEstimate final {
  std::uint64_t parameter_bytes;
  std::uint64_t gradient_bytes;
  std::uint64_t first_moment_bytes;
  std::uint64_t second_moment_bytes;
  std::uint64_t persistent_training_bytes;
};

/// Deterministic persistent-state estimate. Excludes activations, graph metadata, temporaries,
/// allocator overhead, and data batches.
[[nodiscard]] AdamWMemoryEstimate adamw_memory_estimate(Leda& model);

} // namespace spar::leda
