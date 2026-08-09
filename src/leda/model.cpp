module spar.leda.model;

import std;
import spar.leda.config;
import spar.nn.decoder;
import spar.nn.parameter;
import spar.nn.parameters;
import spar.random;
import spar.tensor;

using namespace std;

namespace spar::leda {
namespace {

bool same_config(const nn::DecoderConfig& left, const nn::DecoderConfig& right) noexcept {
  return left.vocab_size == right.vocab_size && left.model_dim == right.model_dim &&
         left.hidden_dim == right.hidden_dim && left.num_layers == right.num_layers &&
         left.num_query_heads == right.num_query_heads && left.num_kv_heads == right.num_kv_heads &&
         left.dtype == right.dtype && left.attention_bias == right.attention_bias &&
         left.mlp_bias == right.mlp_bias && left.norm_epsilon == right.norm_epsilon &&
         left.rope_theta == right.rope_theta && left.qk_norm == right.qk_norm &&
         left.qk_norm_epsilon == right.qk_norm_epsilon;
}

void checked_add(uint64_t& destination, size_t value, string_view description) {
  if (value > numeric_limits<uint64_t>::max() ||
      static_cast<uint64_t>(value) > numeric_limits<uint64_t>::max() - destination) {
    throw overflow_error{string{"Leda "} + string{description} + " overflow"};
  }
  destination += static_cast<uint64_t>(value);
}

void checked_add_u64(uint64_t& destination, uint64_t value, string_view description) {
  if (value > numeric_limits<uint64_t>::max() - destination) {
    throw overflow_error{string{"Leda "} + string{description} + " overflow"};
  }
  destination += value;
}

} // namespace

Leda::Leda(LedaConfig config, Random& random)
    : config_{std::move(config)}, decoder_{decoder_config(config_), random} {}

Leda::Leda(LedaConfig config, nn::DecoderLM decoder, int)
    : config_{std::move(config)}, decoder_{std::move(decoder)} {}

Leda Leda::from_decoder(LedaConfig config, nn::DecoderLM decoder) {
  const nn::DecoderConfig mapped{decoder_config(config)};
  if (!same_config(mapped, decoder.config())) {
    throw invalid_argument{"Checkpoint DecoderConfig does not match LedaConfig"};
  }
  return Leda{std::move(config), std::move(decoder), 0};
}

Tensor Leda::forward(const Tensor& token_ids, size_t start_position) const {
  return decoder_.forward(token_ids, start_position);
}

const LedaConfig& Leda::config() const noexcept {
  return config_;
}

nn::DecoderLM& Leda::decoder() noexcept {
  return decoder_;
}

const nn::DecoderLM& Leda::decoder() const noexcept {
  return decoder_;
}

vector<nn::NamedParameter> named_parameters(Leda& model) {
  return nn::named_parameters(model.decoder());
}

vector<nn::Parameter> parameters(Leda& model) {
  return nn::parameters(model.decoder());
}

void zero_grad(Leda& model) {
  nn::zero_grad(model.decoder());
}

ModelStatistics model_statistics(Leda& model) {
  uint64_t total_parameters{0};
  uint64_t trainable_parameters{0};
  uint64_t parameter_bytes{0};
  vector<nn::Parameter> unique;
  for (nn::Parameter parameter : parameters(model)) {
    const bool duplicate{ranges::any_of(unique, [&parameter](const nn::Parameter& existing) {
      return existing.shares_identity_with(parameter);
    })};
    if (duplicate) {
      continue;
    }
    unique.push_back(parameter);
    checked_add(total_parameters, parameter.tensor().numel(), "parameter count");
    checked_add(parameter_bytes, parameter.tensor().nbytes(), "parameter byte count");
    if (parameter.requires_grad()) {
      checked_add(trainable_parameters, parameter.tensor().numel(), "trainable parameter count");
    }
  }
  return {.total_parameters = total_parameters,
          .trainable_parameters = trainable_parameters,
          .parameter_bytes = parameter_bytes};
}

AdamWMemoryEstimate adamw_memory_estimate(Leda& model) {
  const ModelStatistics statistics{model_statistics(model)};
  uint64_t trainable_bytes{0};
  for (nn::Parameter parameter : parameters(model)) {
    if (parameter.requires_grad()) {
      checked_add(trainable_bytes, parameter.tensor().nbytes(), "trainable byte count");
    }
  }
  uint64_t persistent{statistics.parameter_bytes};
  checked_add_u64(persistent, trainable_bytes, "persistent training byte count");
  checked_add_u64(persistent, trainable_bytes, "persistent training byte count");
  checked_add_u64(persistent, trainable_bytes, "persistent training byte count");
  return {.parameter_bytes = statistics.parameter_bytes,
          .gradient_bytes = trainable_bytes,
          .first_moment_bytes = trainable_bytes,
          .second_moment_bytes = trainable_bytes,
          .persistent_training_bytes = persistent};
}

} // namespace spar::leda
