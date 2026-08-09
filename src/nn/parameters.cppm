export module spar.nn.parameters;

import std;
export import spar.nn.attention;
export import spar.nn.embedding;
export import spar.nn.linear;
export import spar.nn.mlp;
export import spar.nn.parameter;
export import spar.nn.rmsnorm;
export import spar.nn.transformer;

export namespace spar::nn {

struct NamedParameter final {
  std::string name;
  Parameter parameter;
};

[[nodiscard]] std::vector<NamedParameter> named_parameters(Linear& layer);
[[nodiscard]] std::vector<NamedParameter> named_parameters(Embedding& layer);
[[nodiscard]] std::vector<NamedParameter> named_parameters(RMSNorm& layer);
[[nodiscard]] std::vector<NamedParameter> named_parameters(SelfAttention& layer);
[[nodiscard]] std::vector<NamedParameter> named_parameters(SwiGLUMLP& layer);
[[nodiscard]] std::vector<NamedParameter> named_parameters(TransformerBlock& layer);

[[nodiscard]] std::vector<Parameter> parameters(Linear& layer);
[[nodiscard]] std::vector<Parameter> parameters(Embedding& layer);
[[nodiscard]] std::vector<Parameter> parameters(RMSNorm& layer);
[[nodiscard]] std::vector<Parameter> parameters(SelfAttention& layer);
[[nodiscard]] std::vector<Parameter> parameters(SwiGLUMLP& layer);
[[nodiscard]] std::vector<Parameter> parameters(TransformerBlock& layer);

void zero_grad(Linear& layer);
void zero_grad(Embedding& layer);
void zero_grad(RMSNorm& layer);
void zero_grad(SelfAttention& layer);
void zero_grad(SwiGLUMLP& layer);
void zero_grad(TransformerBlock& layer);

} // namespace spar::nn
