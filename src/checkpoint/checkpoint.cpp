module spar.checkpoint;

import std;
import spar.dtype;
import spar.nn.decoder;
import spar.nn.parameters;
import spar.nn.state;
import spar.optim.adamw;
import spar.random;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar::checkpoint {
namespace {

constexpr array<char, 8> magic{'S', 'P', 'A', 'R', 'C', 'K', 'P', 'T'};
constexpr uint32_t format_version{1};
constexpr uint32_t endian_marker{0x01020304U};
constexpr uint64_t payload_alignment{64};
constexpr uint32_t maximum_name_bytes{1024U * 1024U};
constexpr uint32_t maximum_rank{64};
constexpr size_t payload_chunk_bytes{1U << 20U};

static_assert(endian::native == endian::little || endian::native == endian::big);
static_assert(sizeof(float) == 4);
static_assert(sizeof(double) == 8);
static_assert(sizeof(int32_t) == 4);
static_assert(sizeof(int64_t) == 8);
static_assert(numeric_limits<float>::is_iec559);
static_assert(numeric_limits<double>::is_iec559);

// Spar checkpoint v1 is a canonical little-endian stream:
// magic/version/endian marker; DecoderConfig; progress; RNG state; AdamW
// hyperparameters; named model records; then name-keyed optimizer records.
// Every tensor payload starts at the next 64-byte boundary and contains
// logical contiguous row-major scalar encodings, never C++ object layouts.

class Writer final {
public:
  explicit Writer(const filesystem::path& path)
      : stream_{path, ios::binary | ios::trunc}, position_{0} {
    if (!stream_) {
      throw runtime_error{"Unable to open checkpoint temporary file for writing"};
    }
  }

  void bytes(const char* data, size_t size) {
    for (size_t offset{0}; offset < size;) {
      const size_t chunk{min(size - offset, payload_chunk_bytes)};
      stream_.write(data + offset, static_cast<streamsize>(chunk));
      if (!stream_) {
        throw runtime_error{"Checkpoint write failed"};
      }
      offset += chunk;
      position_ += chunk;
    }
  }
  void bytes(span<const byte> data) {
    bytes(reinterpret_cast<const char*>(data.data()), data.size());
  }
  void u8(uint8_t value) {
    const char byte{static_cast<char>(value)};
    bytes(&byte, 1);
  }
  void boolean(bool value) {
    u8(value ? 1 : 0);
  }
  void u32(uint32_t value) {
    array<char, 4> data{};
    for (size_t index{0}; index < data.size(); ++index) {
      data[index] = static_cast<char>((value >> (index * 8U)) & 0xffU);
    }
    bytes(data.data(), data.size());
  }
  void u64(uint64_t value) {
    array<char, 8> data{};
    for (size_t index{0}; index < data.size(); ++index) {
      data[index] = static_cast<char>((value >> (index * 8U)) & 0xffU);
    }
    bytes(data.data(), data.size());
  }
  void f64(double value) {
    u64(bit_cast<uint64_t>(value));
  }
  void string_value(string_view value) {
    if (value.size() > maximum_name_bytes) {
      throw invalid_argument{"Checkpoint name exceeds the v1 limit"};
    }
    u32(static_cast<uint32_t>(value.size()));
    bytes(value.data(), value.size());
  }
  void align_payload() {
    const uint64_t padding{(payload_alignment - position_ % payload_alignment) % payload_alignment};
    constexpr array<char, 64> zeros{};
    bytes(zeros.data(), static_cast<size_t>(padding));
  }
  void finish() {
    stream_.flush();
    if (!stream_) {
      throw runtime_error{"Checkpoint flush failed"};
    }
    stream_.close();
    if (!stream_) {
      throw runtime_error{"Checkpoint close failed"};
    }
  }

private:
  ofstream stream_;
  uint64_t position_;
};

class Reader final {
public:
  explicit Reader(const filesystem::path& path) : stream_{path, ios::binary}, position_{0} {
    if (!stream_) {
      throw runtime_error{"Unable to open checkpoint for reading"};
    }
  }

  void bytes(char* destination, size_t size) {
    for (size_t offset{0}; offset < size;) {
      const size_t chunk{min(size - offset, payload_chunk_bytes)};
      stream_.read(destination + offset, static_cast<streamsize>(chunk));
      if (stream_.gcount() != static_cast<streamsize>(chunk)) {
        throw runtime_error{"Checkpoint is truncated"};
      }
      offset += chunk;
      position_ += chunk;
    }
  }
  void bytes(span<byte> destination) {
    bytes(reinterpret_cast<char*>(destination.data()), destination.size());
  }
  uint8_t u8() {
    char byte{};
    bytes(&byte, 1);
    return static_cast<uint8_t>(static_cast<unsigned char>(byte));
  }
  bool boolean() {
    const uint8_t value{u8()};
    if (value > 1) {
      throw runtime_error{"Checkpoint contains an invalid boolean"};
    }
    return value != 0;
  }
  uint32_t u32() {
    array<char, 4> data{};
    bytes(data.data(), data.size());
    uint32_t value{0};
    for (size_t index{0}; index < data.size(); ++index) {
      value |= static_cast<uint32_t>(static_cast<unsigned char>(data[index])) << (index * 8U);
    }
    return value;
  }
  uint64_t u64() {
    array<char, 8> data{};
    bytes(data.data(), data.size());
    uint64_t value{0};
    for (size_t index{0}; index < data.size(); ++index) {
      value |= static_cast<uint64_t>(static_cast<unsigned char>(data[index])) << (index * 8U);
    }
    return value;
  }
  double f64() {
    return bit_cast<double>(u64());
  }
  string string_value() {
    const uint32_t size{u32()};
    if (size > maximum_name_bytes) {
      throw runtime_error{"Checkpoint name exceeds the v1 limit"};
    }
    string value(size, '\0');
    bytes(value.data(), value.size());
    return value;
  }
  void align_payload() {
    const uint64_t padding{(payload_alignment - position_ % payload_alignment) % payload_alignment};
    array<char, 64> data{};
    bytes(data.data(), static_cast<size_t>(padding));
    if (ranges::any_of(span<const char>{data.data(), static_cast<size_t>(padding)},
                       [](char byte) { return byte != 0; })) {
      throw runtime_error{"Checkpoint payload alignment padding is not zero"};
    }
  }
  void require_eof() {
    char extra{};
    stream_.read(&extra, 1);
    if (stream_.gcount() != 0 || !stream_.eof()) {
      throw runtime_error{"Checkpoint has trailing data or an I/O error"};
    }
  }

private:
  ifstream stream_;
  uint64_t position_;
};

uint8_t wire_dtype(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return 1;
  case DType::Float64:
    return 2;
  case DType::Int32:
    return 3;
  case DType::Int64:
    return 4;
  }
  throw invalid_argument{"Unknown Spar dtype"};
}

DType spar_dtype(uint8_t tag) {
  switch (tag) {
  case 1:
    return DType::Float32;
  case 2:
    return DType::Float64;
  case 3:
    return DType::Int32;
  case 4:
    return DType::Int64;
  default:
    throw runtime_error{"Checkpoint contains an unknown dtype tag"};
  }
}

template <typename T, typename Bits>
void write_tensor_values(Writer& writer, const Tensor& tensor) {
  if constexpr (endian::native == endian::little) {
    if (tensor.is_contiguous()) {
      writer.bytes(as_bytes(tensor.span<T>()));
      return;
    }
  }

  const size_t chunk_elements{payload_chunk_bytes / sizeof(Bits)};
  vector<Bits> encoded(min(tensor.numel(), chunk_elements));
  for (size_t offset{0}; offset < tensor.numel(); offset += encoded.size()) {
    const size_t count{min(encoded.size(), tensor.numel() - offset)};
    for (size_t index{0}; index < count; ++index) {
      Bits bits{bit_cast<Bits>(detail::logical_value<T>(tensor, offset + index))};
      if constexpr (endian::native == endian::big) {
        bits = byteswap(bits);
      }
      encoded[index] = bits;
    }
    writer.bytes(as_bytes(span<const Bits>{encoded.data(), count}));
  }
}

void write_payload(Writer& writer, const Tensor& tensor) {
  writer.align_payload();
  switch (tensor.dtype()) {
  case DType::Float32:
    write_tensor_values<float, uint32_t>(writer, tensor);
    break;
  case DType::Float64:
    write_tensor_values<double, uint64_t>(writer, tensor);
    break;
  case DType::Int32:
    write_tensor_values<int32_t, uint32_t>(writer, tensor);
    break;
  case DType::Int64:
    write_tensor_values<int64_t, uint64_t>(writer, tensor);
    break;
  }
}

template <typename T, typename Bits> void read_tensor_values(Reader& reader, Tensor& tensor) {
  auto values{tensor.span<T>()};
  if constexpr (endian::native == endian::little) {
    reader.bytes(as_writable_bytes(values));
    return;
  }

  const size_t chunk_elements{payload_chunk_bytes / sizeof(Bits)};
  vector<Bits> encoded(min(values.size(), chunk_elements));
  for (size_t offset{0}; offset < values.size(); offset += encoded.size()) {
    const size_t count{min(encoded.size(), values.size() - offset)};
    reader.bytes(as_writable_bytes(span<Bits>{encoded.data(), count}));
    for (size_t index{0}; index < count; ++index) {
      values[offset + index] = bit_cast<T>(byteswap(encoded[index]));
    }
  }
}

Tensor read_payload(Reader& reader, Shape shape, DType dtype, uint64_t encoded_nbytes) {
  const uint64_t expected{static_cast<uint64_t>(shape.numel()) * size_of(dtype)};
  if (encoded_nbytes != expected) {
    throw runtime_error{"Checkpoint tensor payload byte count mismatch"};
  }
  reader.align_payload();
  Tensor tensor{std::move(shape), dtype};
  switch (dtype) {
  case DType::Float32:
    read_tensor_values<float, uint32_t>(reader, tensor);
    break;
  case DType::Float64:
    read_tensor_values<double, uint64_t>(reader, tensor);
    break;
  case DType::Int32:
    read_tensor_values<int32_t, uint32_t>(reader, tensor);
    break;
  case DType::Int64:
    read_tensor_values<int64_t, uint64_t>(reader, tensor);
    break;
  }
  return tensor;
}

void write_config(Writer& writer, const nn::DecoderConfig& config) {
  writer.u64(config.vocab_size);
  writer.u64(config.model_dim);
  writer.u64(config.hidden_dim);
  writer.u64(config.num_layers);
  writer.u64(config.num_query_heads);
  writer.u64(config.num_kv_heads);
  writer.u8(wire_dtype(config.dtype));
  writer.boolean(config.attention_bias);
  writer.boolean(config.mlp_bias);
  writer.f64(config.norm_epsilon);
  writer.f64(config.rope_theta);
  writer.boolean(config.qk_norm);
  writer.f64(config.qk_norm_epsilon);
}

size_t checked_size(uint64_t value) {
  if (value > numeric_limits<size_t>::max()) {
    throw runtime_error{"Checkpoint count does not fit this platform"};
  }
  return static_cast<size_t>(value);
}

nn::DecoderConfig read_config(Reader& reader) {
  nn::DecoderConfig config;
  config.vocab_size = checked_size(reader.u64());
  config.model_dim = checked_size(reader.u64());
  config.hidden_dim = checked_size(reader.u64());
  config.num_layers = checked_size(reader.u64());
  config.num_query_heads = checked_size(reader.u64());
  config.num_kv_heads = checked_size(reader.u64());
  config.dtype = spar_dtype(reader.u8());
  config.attention_bias = reader.boolean();
  config.mlp_bias = reader.boolean();
  config.norm_epsilon = reader.f64();
  config.rope_theta = reader.f64();
  config.qk_norm = reader.boolean();
  config.qk_norm_epsilon = reader.f64();
  return config;
}

void write_shape(Writer& writer, const Shape& shape) {
  if (shape.rank() > maximum_rank) {
    throw invalid_argument{"Checkpoint tensor rank exceeds the v1 limit"};
  }
  writer.u32(static_cast<uint32_t>(shape.rank()));
  for (const auto dimension : shape.dimensions()) {
    writer.u64(static_cast<uint64_t>(dimension));
  }
}

Shape read_shape(Reader& reader) {
  const uint32_t rank{reader.u32()};
  if (rank > maximum_rank) {
    throw runtime_error{"Checkpoint tensor rank exceeds the v1 limit"};
  }
  vector<Shape::dimension_type> dimensions;
  dimensions.reserve(rank);
  const auto maximum{static_cast<uint64_t>(numeric_limits<Shape::dimension_type>::max())};
  for (uint32_t index{0}; index < rank; ++index) {
    const uint64_t dimension{reader.u64()};
    if (dimension > maximum) {
      throw runtime_error{"Checkpoint tensor dimension is too large"};
    }
    dimensions.push_back(static_cast<Shape::dimension_type>(dimension));
  }
  return Shape{dimensions};
}

void validate_optimizer_state(const nn::Parameter& parameter,
                              const optim::AdamWParameterState& state) {
  const Tensor& value{parameter.tensor()};
  if (state.step == 0 || state.first_moment.shape() != value.shape() ||
      state.second_moment.shape() != value.shape() || state.first_moment.dtype() != value.dtype() ||
      state.second_moment.dtype() != value.dtype() ||
      (value.dtype() != DType::Float32 && value.dtype() != DType::Float64)) {
    throw invalid_argument{"Checkpoint optimizer state is structurally invalid"};
  }
}

} // namespace

void save_training_checkpoint(const filesystem::path& path, nn::DecoderLM& model,
                              const optim::AdamW& optimizer, const Random& random,
                              TrainingProgress progress) {
  const auto named{nn::named_parameters(model)};
  if (ranges::any_of(named,
                     [](const nn::NamedParameter& entry) { return entry.parameter.has_grad(); })) {
    throw invalid_argument{"Cannot checkpoint a model with accumulated gradients"};
  }
  if (optimizer.parameter_count() != named.size() ||
      ranges::any_of(named, [&optimizer](const nn::NamedParameter& entry) {
        return !optimizer.tracks(entry.parameter);
      })) {
    throw invalid_argument{"Checkpoint optimizer does not track exactly the model Parameters"};
  }
  // Prevalidate one cloned optimizer state at a time before creating the temporary file.
  for (const nn::NamedParameter& entry : named) {
    const auto state{optimizer.parameter_state(entry.parameter)};
    if (state) {
      validate_optimizer_state(entry.parameter, *state);
    }
  }

  filesystem::path temporary{path};
  temporary += ".tmp";
  error_code ignored;
  filesystem::remove(temporary, ignored);
  try {
    Writer writer{temporary};
    writer.bytes(magic.data(), magic.size());
    writer.u32(format_version);
    writer.u32(endian_marker);
    write_config(writer, model.config());
    writer.u64(progress.global_step);
    writer.u64(progress.tokens_seen);
    writer.u64(random.state());
    writer.f64(optimizer.learning_rate());
    writer.f64(optimizer.beta1());
    writer.f64(optimizer.beta2());
    writer.f64(optimizer.epsilon());
    writer.f64(optimizer.weight_decay());
    writer.u64(named.size());
    for (const nn::NamedParameter& entry : named) {
      const Tensor& value{entry.parameter.tensor()};
      writer.string_value(entry.name);
      writer.u8(wire_dtype(value.dtype()));
      writer.boolean(entry.parameter.requires_grad());
      write_shape(writer, value.shape());
      writer.u64(value.nbytes());
      write_payload(writer, value);
    }
    writer.u64(named.size());
    for (const nn::NamedParameter& entry : named) {
      writer.string_value(entry.name);
      const auto state{optimizer.parameter_state(entry.parameter)};
      writer.boolean(state.has_value());
      if (state) {
        writer.u64(state->step);
        writer.u64(state->first_moment.nbytes());
        write_payload(writer, state->first_moment);
        write_payload(writer, state->second_moment);
      }
    }
    writer.finish();
    filesystem::rename(temporary, path);
  } catch (...) {
    filesystem::remove(temporary, ignored);
    throw;
  }
}

LoadedTrainingCheckpoint load_training_checkpoint(const filesystem::path& path) {
  Reader reader{path};
  array<char, 8> encoded_magic{};
  reader.bytes(encoded_magic.data(), encoded_magic.size());
  if (encoded_magic != magic) {
    throw runtime_error{"Invalid Spar checkpoint magic"};
  }
  if (reader.u32() != format_version) {
    throw runtime_error{"Unsupported Spar checkpoint version"};
  }
  if (reader.u32() != endian_marker) {
    throw runtime_error{"Invalid Spar checkpoint endian marker"};
  }
  const nn::DecoderConfig config{read_config(reader)};
  const TrainingProgress progress{reader.u64(), reader.u64()};
  const uint64_t random_state{reader.u64()};
  const double learning_rate{reader.f64()};
  const double beta1{reader.f64()};
  const double beta2{reader.f64()};
  const double epsilon{reader.f64()};
  const double weight_decay{reader.f64()};

  Random construction_random{0};
  nn::DecoderLM model{config, construction_random};
  const auto expected_parameters{nn::named_parameters(model)};
  const uint64_t model_count{reader.u64()};
  if (model_count != expected_parameters.size()) {
    throw runtime_error{"Checkpoint model Parameter count mismatch"};
  }
  vector<nn::NamedTensor> model_state;
  vector<pair<string, bool>> trainability;
  unordered_map<string_view, const nn::Parameter*> expected_by_name;
  for (const nn::NamedParameter& entry : expected_parameters) {
    expected_by_name.emplace(entry.name, &entry.parameter);
  }
  unordered_set<string> model_names;
  model_state.reserve(expected_parameters.size());
  trainability.reserve(expected_parameters.size());
  for (size_t index{0}; index < expected_parameters.size(); ++index) {
    string name{reader.string_value()};
    if (!model_names.insert(name).second) {
      throw runtime_error{"Checkpoint contains a duplicate model Parameter name"};
    }
    const auto expected{expected_by_name.find(name)};
    if (expected == expected_by_name.end()) {
      throw runtime_error{"Checkpoint contains an unknown model Parameter name"};
    }
    const DType dtype{spar_dtype(reader.u8())};
    const bool requires_grad{reader.boolean()};
    Shape shape{read_shape(reader)};
    if (dtype != expected->second->tensor().dtype() ||
        shape != expected->second->tensor().shape()) {
      throw runtime_error{"Checkpoint model Parameter metadata mismatch"};
    }
    const uint64_t nbytes{reader.u64()};
    model_state.push_back({name, read_payload(reader, std::move(shape), dtype, nbytes)});
    trainability.emplace_back(std::move(name), requires_grad);
  }
  nn::load_state_dict(model, model_state);
  unordered_map<string_view, bool> trainability_by_name;
  for (const auto& [name, requires_grad] : trainability) {
    trainability_by_name.emplace(name, requires_grad);
  }
  for (nn::NamedParameter& entry : nn::named_parameters(model)) {
    const auto iterator{trainability_by_name.find(entry.name)};
    if (iterator == trainability_by_name.end()) {
      throw runtime_error{"Checkpoint trainability metadata is incomplete"};
    }
    entry.parameter.set_requires_grad(iterator->second);
  }

  optim::AdamW optimizer{nn::parameters(model), learning_rate, beta1, beta2, epsilon, weight_decay};
  auto named{nn::named_parameters(model)};
  unordered_map<string_view, nn::Parameter*> by_name;
  for (nn::NamedParameter& entry : named) {
    by_name.emplace(entry.name, &entry.parameter);
  }
  const uint64_t optimizer_count{reader.u64()};
  if (optimizer_count != named.size()) {
    throw runtime_error{"Checkpoint optimizer Parameter count mismatch"};
  }
  unordered_set<string> optimizer_names;
  for (size_t index{0}; index < named.size(); ++index) {
    string name{reader.string_value()};
    if (!optimizer_names.insert(name).second) {
      throw runtime_error{"Checkpoint contains a duplicate optimizer Parameter name"};
    }
    const auto parameter_iterator{by_name.find(name)};
    if (parameter_iterator == by_name.end()) {
      throw runtime_error{"Checkpoint optimizer references an unknown Parameter"};
    }
    const bool has_state{reader.boolean()};
    if (!has_state) {
      optimizer.set_parameter_state(*parameter_iterator->second, nullopt);
      continue;
    }
    const uint64_t step{reader.u64()};
    const uint64_t nbytes{reader.u64()};
    const Tensor& parameter_value{parameter_iterator->second->tensor()};
    Tensor first{read_payload(reader, parameter_value.shape(), parameter_value.dtype(), nbytes)};
    Tensor second{read_payload(reader, parameter_value.shape(), parameter_value.dtype(), nbytes)};
    optimizer.set_parameter_state(
        *parameter_iterator->second,
        optim::AdamWParameterState{std::move(first), std::move(second), step});
  }
  reader.require_eof();
  Random random{0};
  random.set_state(random_state);
  return LoadedTrainingCheckpoint{std::move(model), std::move(optimizer), std::move(random),
                                  progress};
}

} // namespace spar::checkpoint
