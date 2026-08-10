module spar.nn.state;

import std;
import spar.dtype;
import spar.nn.decoder;
import spar.nn.parameters;
import spar.tensor;

using namespace std;

namespace spar::nn {
vector<NamedTensor> state_dict(DecoderLM& model) {
  vector<NamedTensor> result;
  const auto named{named_parameters(model)};
  result.reserve(named.size());
  for (const NamedParameter& entry : named) {
    result.push_back({entry.name, entry.parameter.tensor().detach().clone()});
  }
  return result;
}

void load_state_dict(DecoderLM& model, span<const NamedTensor> state) {
  const auto destination{named_parameters(model)};
  unordered_map<string_view, const NamedTensor*> incoming;
  incoming.reserve(state.size());
  for (const NamedTensor& entry : state) {
    if (!incoming.emplace(entry.name, &entry).second) {
      throw invalid_argument{"load_state_dict received a duplicate name"};
    }
  }
  if (incoming.size() != destination.size()) {
    throw invalid_argument{"load_state_dict name count mismatch"};
  }
  for (const NamedParameter& expected : destination) {
    const auto iterator{incoming.find(expected.name)};
    if (iterator == incoming.end()) {
      throw invalid_argument{"load_state_dict is missing a Parameter"};
    }
    const Tensor& value{iterator->second->value};
    if (value.shape() != expected.parameter.tensor().shape()) {
      throw invalid_argument{"load_state_dict Parameter shape mismatch"};
    }
    if (value.dtype() != expected.parameter.tensor().dtype()) {
      throw invalid_argument{"load_state_dict Parameter dtype mismatch"};
    }
    if (value.device() != expected.parameter.tensor().device()) {
      throw invalid_argument{"load_state_dict Parameter Device mismatch"};
    }
  }
  for (const NamedParameter& expected : destination) {
    Tensor target{expected.parameter.tensor().detach()};
    detail::copy_tensor_values(target, incoming.at(expected.name)->value);
  }
}

} // namespace spar::nn
