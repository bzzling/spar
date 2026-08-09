module spar.nn.parameters;

import std;
import spar.nn.attention;
import spar.nn.embedding;
import spar.nn.linear;
import spar.nn.mlp;
import spar.nn.parameter;
import spar.nn.rmsnorm;
import spar.nn.transformer;

using namespace std;

namespace spar::nn {
namespace {

void append_unique(vector<NamedParameter>& destination, string name, Parameter parameter) {
  const bool duplicate{ranges::any_of(destination, [&parameter](const NamedParameter& existing) {
    return existing.parameter.shares_identity_with(parameter);
  })};
  if (!duplicate) {
    destination.push_back(NamedParameter{std::move(name), std::move(parameter)});
  }
}

void append_prefixed(vector<NamedParameter>& destination, string_view prefix,
                     vector<NamedParameter> source) {
  for (auto& named : source) {
    append_unique(destination, string{prefix} + named.name, std::move(named.parameter));
  }
}

vector<Parameter> unnamed(vector<NamedParameter> named) {
  vector<Parameter> result;
  result.reserve(named.size());
  for (auto& entry : named) {
    result.push_back(std::move(entry.parameter));
  }
  return result;
}

template <typename Layer> void clear_gradients(Layer& layer) {
  for (Parameter& parameter : parameters(layer)) {
    parameter.zero_grad();
  }
}

} // namespace

vector<NamedParameter> named_parameters(Linear& layer) {
  vector<NamedParameter> result;
  result.reserve(layer.has_bias() ? 2 : 1);
  append_unique(result, "weight", layer.weight());
  if (Parameter * bias{layer.bias()}) {
    append_unique(result, "bias", *bias);
  }
  return result;
}

vector<NamedParameter> named_parameters(Embedding& layer) {
  return {{"weight", layer.weight()}};
}

vector<NamedParameter> named_parameters(RMSNorm& layer) {
  return {{"weight", layer.weight()}};
}

vector<NamedParameter> named_parameters(SelfAttention& layer) {
  vector<NamedParameter> result;
  append_prefixed(result, "q_proj.", named_parameters(layer.q_proj()));
  append_prefixed(result, "k_proj.", named_parameters(layer.k_proj()));
  append_prefixed(result, "v_proj.", named_parameters(layer.v_proj()));
  append_prefixed(result, "out_proj.", named_parameters(layer.out_proj()));
  if (RMSNorm * q_norm{layer.q_norm()}) {
    append_prefixed(result, "q_norm.", named_parameters(*q_norm));
  }
  if (RMSNorm * k_norm{layer.k_norm()}) {
    append_prefixed(result, "k_norm.", named_parameters(*k_norm));
  }
  return result;
}

vector<NamedParameter> named_parameters(SwiGLUMLP& layer) {
  vector<NamedParameter> result;
  append_prefixed(result, "gate_proj.", named_parameters(layer.gate_proj()));
  append_prefixed(result, "up_proj.", named_parameters(layer.up_proj()));
  append_prefixed(result, "down_proj.", named_parameters(layer.down_proj()));
  return result;
}

vector<NamedParameter> named_parameters(TransformerBlock& layer) {
  vector<NamedParameter> result;
  append_prefixed(result, "attention_norm.", named_parameters(layer.attention_norm()));
  append_prefixed(result, "attention.", named_parameters(layer.attention()));
  append_prefixed(result, "mlp_norm.", named_parameters(layer.mlp_norm()));
  append_prefixed(result, "mlp.", named_parameters(layer.mlp()));
  return result;
}

vector<Parameter> parameters(Linear& layer) {
  return unnamed(named_parameters(layer));
}
vector<Parameter> parameters(Embedding& layer) {
  return unnamed(named_parameters(layer));
}
vector<Parameter> parameters(RMSNorm& layer) {
  return unnamed(named_parameters(layer));
}
vector<Parameter> parameters(SelfAttention& layer) {
  return unnamed(named_parameters(layer));
}
vector<Parameter> parameters(SwiGLUMLP& layer) {
  return unnamed(named_parameters(layer));
}
vector<Parameter> parameters(TransformerBlock& layer) {
  return unnamed(named_parameters(layer));
}

void zero_grad(Linear& layer) {
  clear_gradients(layer);
}
void zero_grad(Embedding& layer) {
  clear_gradients(layer);
}
void zero_grad(RMSNorm& layer) {
  clear_gradients(layer);
}
void zero_grad(SelfAttention& layer) {
  clear_gradients(layer);
}
void zero_grad(SwiGLUMLP& layer) {
  clear_gradients(layer);
}
void zero_grad(TransformerBlock& layer) {
  clear_gradients(layer);
}

} // namespace spar::nn
