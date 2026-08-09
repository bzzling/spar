export module spar.nn.state;

import std;
export import spar.nn.decoder;
export import spar.nn.parameters;
export import spar.tensor;

export namespace spar::nn {

struct NamedTensor final {
  std::string name;
  Tensor value;
};

[[nodiscard]] std::vector<NamedTensor> state_dict(DecoderLM& model);
void load_state_dict(DecoderLM& model, std::span<const NamedTensor> state);

} // namespace spar::nn
