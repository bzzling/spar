module spar.nn.embedding;

import std;
import spar.dtype;
import spar.nn.parameter;
import spar.ops.embedding;
import spar.random;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar::nn {
namespace {

Tensor make_embedding_weight(size_t num_embeddings, size_t embedding_dim, DType dtype,
                             Random& random) {
  if (num_embeddings == 0 || embedding_dim == 0) {
    throw invalid_argument{"Embedding dimensions must be positive"};
  }
  if (dtype != DType::Float32 && dtype != DType::Float64) {
    throw invalid_argument{"Embedding requires a floating-point dtype"};
  }
  const auto maximum_dimension{static_cast<size_t>(numeric_limits<Shape::dimension_type>::max())};
  if (num_embeddings > maximum_dimension || embedding_dim > maximum_dimension) {
    throw overflow_error{"Embedding dimension exceeds the Shape representation"};
  }
  const double bound{1.0 / sqrt(static_cast<double>(embedding_dim))};
  return random_uniform(Shape{static_cast<Shape::dimension_type>(num_embeddings),
                              static_cast<Shape::dimension_type>(embedding_dim)},
                        dtype, random, -bound, bound);
}

} // namespace

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim, DType dtype, Random& random)
    : num_embeddings_{num_embeddings}, embedding_dim_{embedding_dim},
      weight_{make_embedding_weight(num_embeddings, embedding_dim, dtype, random)} {}

Tensor Embedding::forward(const Tensor& indices) const {
  return embedding_lookup(weight_.tensor(), indices);
}

size_t Embedding::num_embeddings() const noexcept {
  return num_embeddings_;
}

size_t Embedding::embedding_dim() const noexcept {
  return embedding_dim_;
}

Parameter& Embedding::weight() noexcept {
  return weight_;
}

const Parameter& Embedding::weight() const noexcept {
  return weight_;
}

} // namespace spar::nn
