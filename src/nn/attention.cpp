module spar.nn.attention;

import std;
import spar.dtype;
import spar.nn.linear;
import spar.nn.rope;
import spar.ops.elementwise;
import spar.ops.matmul;
import spar.ops.scalar;
import spar.ops.softmax;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar::nn {
namespace {

struct AttentionShape final {
  size_t batch;
  size_t query_heads;
  size_t kv_heads;
  size_t query_length;
  size_t key_length;
  size_t head_dimension;
};

AttentionShape validate_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                  bool causal) {
  if (query.rank() != 4 || key.rank() != 4 || value.rank() != 4) {
    throw invalid_argument{"scaled_dot_product_attention requires rank-4 Q, K, and V"};
  }
  if (query.dtype() != key.dtype() || query.dtype() != value.dtype()) {
    throw invalid_argument{"scaled_dot_product_attention requires identical dtypes"};
  }
  if (query.dtype() != DType::Float32 && query.dtype() != DType::Float64) {
    throw invalid_argument{"scaled_dot_product_attention requires a floating-point dtype"};
  }
  for (const Tensor* tensor : {&query, &key, &value}) {
    if (ranges::any_of(tensor->shape().dimensions(), [](auto extent) { return extent <= 0; })) {
      throw invalid_argument{"scaled_dot_product_attention does not support empty dimensions"};
    }
  }
  if (query.shape()[0] != key.shape()[0] || query.shape()[0] != value.shape()[0]) {
    throw invalid_argument{"scaled_dot_product_attention batch dimensions must match"};
  }
  if (query.shape()[3] != key.shape()[3] || query.shape()[3] != value.shape()[3]) {
    throw invalid_argument{"scaled_dot_product_attention head dimensions must match"};
  }
  if (key.shape()[1] != value.shape()[1]) {
    throw invalid_argument{"scaled_dot_product_attention K/V head counts must match"};
  }
  if (key.shape()[2] != value.shape()[2]) {
    throw invalid_argument{"scaled_dot_product_attention K/V sequence lengths must match"};
  }
  const size_t query_heads{static_cast<size_t>(query.shape()[1])};
  const size_t kv_heads{static_cast<size_t>(key.shape()[1])};
  if (query_heads < kv_heads || query_heads % kv_heads != 0) {
    throw invalid_argument{
        "scaled_dot_product_attention query heads must be divisible by KV heads"};
  }
  if (causal && query.shape()[2] != key.shape()[2]) {
    throw invalid_argument{"causal scaled_dot_product_attention requires equal sequence lengths"};
  }
  return AttentionShape{static_cast<size_t>(query.shape()[0]),
                        query_heads,
                        kv_heads,
                        static_cast<size_t>(query.shape()[2]),
                        static_cast<size_t>(key.shape()[2]),
                        static_cast<size_t>(query.shape()[3])};
}

Tensor repeat_kv_heads(const Tensor& input, size_t query_heads) {
  const size_t batch{static_cast<size_t>(input.shape()[0])};
  const size_t kv_heads{static_cast<size_t>(input.shape()[1])};
  const size_t sequence{static_cast<size_t>(input.shape()[2])};
  const size_t features{static_cast<size_t>(input.shape()[3])};
  if (query_heads == kv_heads) {
    return input;
  }
  const size_t group_size{query_heads / kv_heads};
  const auto dimension = [](size_t value) { return static_cast<Shape::dimension_type>(value); };
  const Tensor grouped{input.contiguous().reshape(
      Shape{dimension(batch), dimension(kv_heads), 1, dimension(sequence), dimension(features)})};
  const Tensor repeated{
      grouped.expand(Shape{dimension(batch), dimension(kv_heads), dimension(group_size),
                           dimension(sequence), dimension(features)})};
  return repeated.contiguous().reshape(
      Shape{dimension(batch), dimension(query_heads), dimension(sequence), dimension(features)});
}

Tensor causal_mask(size_t sequence_length, DType dtype) {
  const auto dimension{static_cast<Shape::dimension_type>(sequence_length)};
  Tensor mask{zeros(Shape{dimension, dimension}, dtype)};
  if (dtype == DType::Float32) {
    auto values{mask.span<float>()};
    for (size_t row{0}; row < sequence_length; ++row) {
      for (size_t column{row + 1}; column < sequence_length; ++column) {
        values[row * sequence_length + column] = -numeric_limits<float>::infinity();
      }
    }
  } else {
    auto values{mask.span<double>()};
    for (size_t row{0}; row < sequence_length; ++row) {
      for (size_t column{row + 1}; column < sequence_length; ++column) {
        values[row * sequence_length + column] = -numeric_limits<double>::infinity();
      }
    }
  }
  return mask;
}

} // namespace

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    bool causal) {
  const AttentionShape shape{validate_attention(query, key, value, causal)};
  const Tensor repeated_key{repeat_kv_heads(key, shape.query_heads)};
  const Tensor repeated_value{repeat_kv_heads(value, shape.query_heads)};
  const Tensor transposed_key{repeated_key.permute({0, 1, 3, 2})};
  Tensor scores{multiply_scalar(matmul(query, transposed_key),
                                1.0 / sqrt(static_cast<double>(shape.head_dimension)))};
  if (causal) {
    scores = add(scores, causal_mask(shape.query_length, query.dtype()));
  }
  const Tensor probabilities{softmax(scores, 3)};
  return matmul(probabilities, repeated_value);
}

SelfAttention::Configuration SelfAttention::validate_configuration(size_t model_dim,
                                                                   size_t num_query_heads,
                                                                   size_t num_kv_heads, DType dtype,
                                                                   double rope_theta) {
  if (model_dim == 0 || num_query_heads == 0 || num_kv_heads == 0) {
    throw invalid_argument{"SelfAttention dimensions must be positive"};
  }
  if (dtype != DType::Float32 && dtype != DType::Float64) {
    throw invalid_argument{"SelfAttention requires a floating-point dtype"};
  }
  if (model_dim % num_query_heads != 0) {
    throw invalid_argument{"SelfAttention model_dim must be divisible by query heads"};
  }
  if (num_query_heads % num_kv_heads != 0) {
    throw invalid_argument{"SelfAttention query heads must be divisible by KV heads"};
  }
  const size_t head_dim{model_dim / num_query_heads};
  if (head_dim % 2 != 0) {
    throw invalid_argument{"SelfAttention head dimension must be even for RoPE"};
  }
  if (!isfinite(rope_theta) || rope_theta <= 0.0) {
    throw invalid_argument{"SelfAttention rope_theta must be finite and positive"};
  }
  return Configuration{model_dim, num_query_heads, num_kv_heads, head_dim, rope_theta};
}

SelfAttention::SelfAttention(size_t model_dim, size_t num_query_heads, size_t num_kv_heads,
                             DType dtype, Random& random, bool bias, double rope_theta)
    : SelfAttention{
          validate_configuration(model_dim, num_query_heads, num_kv_heads, dtype, rope_theta),
          dtype, random, bias} {}

SelfAttention::SelfAttention(Configuration configuration, DType dtype, Random& random, bool bias)
    : model_dim_{configuration.model_dim}, num_query_heads_{configuration.num_query_heads},
      num_kv_heads_{configuration.num_kv_heads}, head_dim_{configuration.head_dim},
      rope_theta_{configuration.rope_theta}, q_proj_{model_dim_, model_dim_, dtype, random, bias},
      k_proj_{model_dim_, num_kv_heads_ * head_dim_, dtype, random, bias},
      v_proj_{model_dim_, num_kv_heads_ * head_dim_, dtype, random, bias},
      out_proj_{model_dim_, model_dim_, dtype, random, bias} {}

Tensor SelfAttention::forward(const Tensor& input, size_t start_position) const {
  if (input.rank() != 3) {
    throw invalid_argument{"SelfAttention input must have shape [B,T,model_dim]"};
  }
  if (input.shape()[0] <= 0 || input.shape()[1] <= 0) {
    throw invalid_argument{"SelfAttention does not support empty batches or sequences"};
  }
  if (static_cast<size_t>(input.shape()[2]) != model_dim_) {
    throw invalid_argument{"SelfAttention input final dimension does not match model_dim"};
  }
  if (input.dtype() != q_proj_.weight().tensor().dtype()) {
    throw invalid_argument{"SelfAttention input dtype does not match its projections"};
  }
  const auto dimension = [](size_t value) { return static_cast<Shape::dimension_type>(value); };
  const auto batch{input.shape()[0]};
  const auto sequence{input.shape()[1]};
  const Tensor query{apply_rope(
      q_proj_.forward(input)
          .reshape(Shape{batch, sequence, dimension(num_query_heads_), dimension(head_dim_)})
          .permute({0, 2, 1, 3}),
      start_position, rope_theta_)};
  const Tensor key{apply_rope(
      k_proj_.forward(input)
          .reshape(Shape{batch, sequence, dimension(num_kv_heads_), dimension(head_dim_)})
          .permute({0, 2, 1, 3}),
      start_position, rope_theta_)};
  const Tensor value{
      v_proj_.forward(input)
          .reshape(Shape{batch, sequence, dimension(num_kv_heads_), dimension(head_dim_)})
          .permute({0, 2, 1, 3})};
  const Tensor context{scaled_dot_product_attention(query, key, value, true)};
  const Tensor merged{context.permute({0, 2, 1, 3})
                          .contiguous()
                          .reshape(Shape{batch, sequence, dimension(model_dim_)})};
  return out_proj_.forward(merged);
}

size_t SelfAttention::model_dim() const noexcept {
  return model_dim_;
}
size_t SelfAttention::num_query_heads() const noexcept {
  return num_query_heads_;
}
size_t SelfAttention::num_kv_heads() const noexcept {
  return num_kv_heads_;
}
size_t SelfAttention::head_dim() const noexcept {
  return head_dim_;
}
double SelfAttention::rope_theta() const noexcept {
  return rope_theta_;
}
Linear& SelfAttention::q_proj() noexcept {
  return q_proj_;
}
const Linear& SelfAttention::q_proj() const noexcept {
  return q_proj_;
}
Linear& SelfAttention::k_proj() noexcept {
  return k_proj_;
}
const Linear& SelfAttention::k_proj() const noexcept {
  return k_proj_;
}
Linear& SelfAttention::v_proj() noexcept {
  return v_proj_;
}
const Linear& SelfAttention::v_proj() const noexcept {
  return v_proj_;
}
Linear& SelfAttention::out_proj() noexcept {
  return out_proj_;
}
const Linear& SelfAttention::out_proj() const noexcept {
  return out_proj_;
}

} // namespace spar::nn
