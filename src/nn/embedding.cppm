export module spar.nn.embedding;

import std;
export import spar.dtype;
export import spar.nn.parameter;
export import spar.random;
export import spar.tensor;

export namespace spar::nn {

class Embedding {
public:
  Embedding(std::size_t num_embeddings, std::size_t embedding_dim, DType dtype, Random& random);

  [[nodiscard]] Tensor forward(const Tensor& indices) const;

  [[nodiscard]] std::size_t num_embeddings() const noexcept;
  [[nodiscard]] std::size_t embedding_dim() const noexcept;
  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

private:
  std::size_t num_embeddings_;
  std::size_t embedding_dim_;
  Parameter weight_;
};

} // namespace spar::nn
