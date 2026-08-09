module spar.nn.decoder;

import std;
import spar.dtype;
import spar.nn.embedding;
import spar.nn.rmsnorm;
import spar.nn.transformer;
import spar.ops.matmul;
import spar.random;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar::nn {

DecoderLM::ValidatedConfiguration DecoderLM::validate_configuration(DecoderConfig config) {
  if (config.vocab_size == 0 || config.model_dim == 0 || config.hidden_dim == 0 ||
      config.num_layers == 0 || config.num_query_heads == 0 || config.num_kv_heads == 0) {
    throw invalid_argument{"DecoderLM dimensions, layer count, and head counts must be positive"};
  }
  if (config.dtype != DType::Float32 && config.dtype != DType::Float64) {
    throw invalid_argument{"DecoderLM requires a floating-point dtype"};
  }
  if (!isfinite(config.norm_epsilon) || config.norm_epsilon <= 0.0) {
    throw invalid_argument{"DecoderLM norm_epsilon must be finite and positive"};
  }
  if (!isfinite(config.rope_theta) || config.rope_theta <= 0.0) {
    throw invalid_argument{"DecoderLM rope_theta must be finite and positive"};
  }
  if (!isfinite(config.qk_norm_epsilon) || config.qk_norm_epsilon <= 0.0) {
    throw invalid_argument{"DecoderLM qk_norm_epsilon must be finite and positive"};
  }
  if (config.model_dim % config.num_query_heads != 0) {
    throw invalid_argument{"DecoderLM model_dim must be divisible by query heads"};
  }
  if (config.num_query_heads % config.num_kv_heads != 0) {
    throw invalid_argument{"DecoderLM query heads must be divisible by KV heads"};
  }
  const size_t head_dim{config.model_dim / config.num_query_heads};
  if (head_dim % 2 != 0) {
    throw invalid_argument{"DecoderLM head dimension must be even for RoPE"};
  }
  const auto maximum_dimension{static_cast<size_t>(numeric_limits<Shape::dimension_type>::max())};
  if (config.vocab_size > maximum_dimension || config.model_dim > maximum_dimension ||
      config.hidden_dim > maximum_dimension) {
    throw overflow_error{"DecoderLM dimension exceeds the Shape representation"};
  }
  return ValidatedConfiguration{std::move(config)};
}

vector<TransformerBlock> DecoderLM::make_blocks(const DecoderConfig& config, Random& random) {
  vector<TransformerBlock> blocks;
  blocks.reserve(config.num_layers);
  for (size_t index{0}; index < config.num_layers; ++index) {
    blocks.emplace_back(config.model_dim, config.hidden_dim, config.num_query_heads,
                        config.num_kv_heads, config.dtype, random, config.attention_bias,
                        config.mlp_bias, config.norm_epsilon, config.rope_theta, config.qk_norm,
                        config.qk_norm_epsilon);
  }
  return blocks;
}

DecoderLM::DecoderLM(DecoderConfig config, Random& random)
    : DecoderLM{validate_configuration(std::move(config)), random} {}

DecoderLM::DecoderLM(ValidatedConfiguration configuration, Random& random)
    : config_{std::move(configuration.value)},
      token_embedding_{config_.vocab_size, config_.model_dim, config_.dtype, random},
      blocks_{make_blocks(config_, random)},
      final_norm_{config_.model_dim, config_.dtype, config_.norm_epsilon} {}

Tensor DecoderLM::forward(const Tensor& token_ids, size_t start_position) const {
  if (token_ids.rank() != 2) {
    throw invalid_argument{"DecoderLM token IDs must have shape [B,T]"};
  }
  if (token_ids.shape()[0] <= 0 || token_ids.shape()[1] <= 0) {
    throw invalid_argument{"DecoderLM does not support empty batches or sequences"};
  }
  if (token_ids.dtype() != DType::Int32 && token_ids.dtype() != DType::Int64) {
    throw invalid_argument{"DecoderLM token IDs must have Int32 or Int64 dtype"};
  }

  Tensor hidden{token_embedding_.forward(token_ids)};
  for (const TransformerBlock& transformer_block : blocks_) {
    hidden = transformer_block.forward(hidden, start_position);
  }
  hidden = final_norm_.forward(hidden);
  return matmul(hidden, token_embedding_.weight().tensor().transpose(0, 1));
}

const DecoderConfig& DecoderLM::config() const noexcept {
  return config_;
}
size_t DecoderLM::vocab_size() const noexcept {
  return config_.vocab_size;
}
size_t DecoderLM::model_dim() const noexcept {
  return config_.model_dim;
}
size_t DecoderLM::num_layers() const noexcept {
  return config_.num_layers;
}
Embedding& DecoderLM::token_embedding() noexcept {
  return token_embedding_;
}
const Embedding& DecoderLM::token_embedding() const noexcept {
  return token_embedding_;
}
TransformerBlock& DecoderLM::block(size_t index) {
  return blocks_.at(index);
}
const TransformerBlock& DecoderLM::block(size_t index) const {
  return blocks_.at(index);
}
span<TransformerBlock> DecoderLM::blocks() noexcept {
  return blocks_;
}
span<const TransformerBlock> DecoderLM::blocks() const noexcept {
  return blocks_;
}
RMSNorm& DecoderLM::final_norm() noexcept {
  return final_norm_;
}
const RMSNorm& DecoderLM::final_norm() const noexcept {
  return final_norm_;
}

} // namespace spar::nn
