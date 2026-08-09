module spar.tokenizer.artifact;

import std;
import spar.tokenizer.byte_bpe;

using namespace std;

namespace spar::tokenizer {
namespace {

constexpr array<char, 8> magic{'S', 'P', 'A', 'R', 'T', 'O', 'K', 'N'};
constexpr uint32_t format_version{1};
constexpr uint32_t base_vocab_size{256};
constexpr uint64_t header_bytes{24};
constexpr uint64_t merge_bytes{8};

class Writer final {
public:
  explicit Writer(const filesystem::path& path) : stream_{path, ios::binary | ios::trunc} {
    if (!stream_) {
      throw runtime_error{"Unable to open tokenizer temporary file for writing"};
    }
  }

  void bytes(const char* data, size_t size) {
    stream_.write(data, static_cast<streamsize>(size));
    if (!stream_) {
      throw runtime_error{"Tokenizer artifact write failed"};
    }
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
  void finish() {
    stream_.flush();
    if (!stream_) {
      throw runtime_error{"Tokenizer artifact flush failed"};
    }
    stream_.close();
    if (!stream_) {
      throw runtime_error{"Tokenizer artifact close failed"};
    }
  }

private:
  ofstream stream_;
};

class Reader final {
public:
  explicit Reader(const filesystem::path& path) : stream_{path, ios::binary} {
    if (!stream_) {
      throw runtime_error{"Unable to open tokenizer artifact"};
    }
  }

  void bytes(char* destination, size_t size) {
    stream_.read(destination, static_cast<streamsize>(size));
    if (stream_.gcount() != static_cast<streamsize>(size)) {
      throw runtime_error{"Tokenizer artifact is truncated"};
    }
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
  void require_eof() {
    char extra{};
    stream_.read(&extra, 1);
    if (stream_.gcount() != 0 || !stream_.eof()) {
      throw runtime_error{"Tokenizer artifact has trailing data or an I/O error"};
    }
  }

private:
  ifstream stream_;
};

} // namespace

void save_tokenizer(const filesystem::path& path, const ByteBPETokenizer& tokenizer) {
  filesystem::path temporary{path};
  temporary += ".tmp";
  error_code ignored;
  filesystem::remove(temporary, ignored);
  try {
    Writer writer{temporary};
    writer.bytes(magic.data(), magic.size());
    writer.u32(format_version);
    writer.u32(base_vocab_size);
    writer.u64(tokenizer.merge_count());
    for (const BPEMerge merge : tokenizer.merges()) {
      writer.u32(merge.left);
      writer.u32(merge.right);
    }
    writer.finish();
    filesystem::rename(temporary, path);
  } catch (...) {
    filesystem::remove(temporary, ignored);
    throw;
  }
}

ByteBPETokenizer load_tokenizer(const filesystem::path& path) {
  error_code size_error;
  const uintmax_t file_bytes{filesystem::file_size(path, size_error)};
  if (size_error) {
    throw runtime_error{"Unable to determine tokenizer artifact size"};
  }
  Reader reader{path};
  array<char, 8> encoded_magic{};
  reader.bytes(encoded_magic.data(), encoded_magic.size());
  if (encoded_magic != magic) {
    throw runtime_error{"Invalid Spar tokenizer artifact magic"};
  }
  if (reader.u32() != format_version) {
    throw runtime_error{"Unsupported Spar tokenizer artifact version"};
  }
  if (reader.u32() != base_vocab_size) {
    throw runtime_error{"Tokenizer artifact base vocabulary must contain 256 bytes"};
  }
  const uint64_t merge_count{reader.u64()};
  constexpr uint64_t maximum_merge_count{static_cast<uint64_t>(numeric_limits<TokenId>::max()) +
                                         1U - base_vocab_size};
  if (merge_count > maximum_merge_count ||
      merge_count > (numeric_limits<uint64_t>::max() - header_bytes) / merge_bytes) {
    throw runtime_error{"Tokenizer artifact merge count exceeds the TokenId range"};
  }
  const uint64_t expected_bytes{header_bytes + merge_count * merge_bytes};
  if (file_bytes != expected_bytes) {
    throw runtime_error{"Tokenizer artifact size does not match its merge count"};
  }
  if (merge_count > static_cast<uint64_t>(vector<BPEMerge>{}.max_size())) {
    throw runtime_error{"Tokenizer artifact merge count is too large for this platform"};
  }

  vector<BPEMerge> merges;
  merges.reserve(static_cast<size_t>(merge_count));
  for (uint64_t rank{0}; rank < merge_count; ++rank) {
    merges.push_back({reader.u32(), reader.u32()});
  }
  reader.require_eof();
  return ByteBPETokenizer{std::move(merges)};
}

} // namespace spar::tokenizer
