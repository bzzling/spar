export module spar.ops.embedding;

export import spar.tensor;

export namespace spar {

/// Selects rows from a floating rank-2 weight using signed integer indices.
[[nodiscard]] Tensor embedding_lookup(const Tensor& weight, const Tensor& indices);

} // namespace spar
