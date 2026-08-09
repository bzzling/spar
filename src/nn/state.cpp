module spar.nn.state;

import std;
import spar.dtype;
import spar.nn.decoder;
import spar.nn.parameters;
import spar.tensor;

using namespace std;

namespace spar::nn {
namespace {

template <typename T> void copy_logical(Tensor& destination, const Tensor& source) {
  auto output{destination.span<T>()};
  for (size_t index{0}; index < output.size(); ++index) {
    output[index] = detail::logical_value<T>(source, index);
  }
}

void copy_tensor(Tensor& destination, const Tensor& source) {
  switch (destination.dtype()) {
  case DType::Float32:
    copy_logical<float>(destination, source);
    break;
  case DType::Float64:
    copy_logical<double>(destination, source);
    break;
  case DType::Int32:
    copy_logical<int32_t>(destination, source);
    break;
  case DType::Int64:
    copy_logical<int64_t>(destination, source);
    break;
  }
}

} // namespace

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
  }
  for (const NamedParameter& expected : destination) {
    Tensor target{expected.parameter.tensor().detach()};
    copy_tensor(target, incoming.at(expected.name)->value);
  }
}

} // namespace spar::nn
