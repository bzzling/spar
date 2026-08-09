module spar.tokenizer.byte_bpe;

import std;

using namespace std;

namespace spar::tokenizer {
namespace {

constexpr size_t base_vocab_size{256};
// Bounds hostile or unreasonable merge-DAG expansion; it is not a normal BPE modeling limit.
constexpr size_t maximum_token_bytes{16U * 1024U * 1024U};
constexpr size_t no_node{numeric_limits<size_t>::max()};

uint64_t pair_key(TokenId left, TokenId right) noexcept {
  return (static_cast<uint64_t>(left) << 32U) | static_cast<uint64_t>(right);
}

pair<TokenId, TokenId> unpack_pair(uint64_t key) noexcept {
  return {static_cast<TokenId>(key >> 32U), static_cast<TokenId>(key)};
}

vector<TokenId> byte_tokens(string_view bytes) {
  vector<TokenId> tokens;
  tokens.reserve(bytes.size());
  for (const char value : bytes) {
    tokens.push_back(static_cast<TokenId>(static_cast<unsigned char>(value)));
  }
  return tokens;
}

struct Node final {
  TokenId token;
  size_t previous;
  size_t next;
  bool alive;
};

struct Candidate final {
  size_t rank;
  size_t left;
  size_t right;
};

struct CandidateLater final {
  bool operator()(const Candidate& left, const Candidate& right) const noexcept {
    if (left.rank != right.rank) {
      return left.rank > right.rank;
    }
    if (left.left != right.left) {
      return left.left > right.left;
    }
    return left.right > right.right;
  }
};

void validate_training_config(BPETrainingConfig config) {
  constexpr uint64_t maximum_vocab_size{static_cast<uint64_t>(numeric_limits<TokenId>::max()) + 1U};
  if (config.target_vocab_size < base_vocab_size ||
      static_cast<uint64_t>(config.target_vocab_size) > maximum_vocab_size) {
    throw invalid_argument{"BPE target vocabulary size is outside the TokenId range"};
  }
  if (config.min_pair_frequency == 0) {
    throw invalid_argument{"BPE minimum pair frequency must be at least one"};
  }
}

} // namespace

ByteBPETokenizer::ByteBPETokenizer() : ByteBPETokenizer{vector<BPEMerge>{}} {}

ByteBPETokenizer::ByteBPETokenizer(vector<BPEMerge> merges) : merges_{std::move(merges)} {
  constexpr uint64_t maximum_merge_count{static_cast<uint64_t>(numeric_limits<TokenId>::max()) +
                                         1U - base_vocab_size};
  if (static_cast<uint64_t>(merges_.size()) > maximum_merge_count) {
    throw invalid_argument{"BPE merge table exceeds the TokenId range"};
  }
  if (merges_.size() > numeric_limits<size_t>::max() - base_vocab_size) {
    throw invalid_argument{"BPE vocabulary size overflows this platform"};
  }

  pieces_.reserve(base_vocab_size + merges_.size());
  for (size_t byte_value{0}; byte_value < base_vocab_size; ++byte_value) {
    pieces_.emplace_back(1, static_cast<char>(static_cast<unsigned char>(byte_value)));
  }
  merge_ranks_.reserve(merges_.size());
  for (size_t rank{0}; rank < merges_.size(); ++rank) {
    const uint64_t output_value{base_vocab_size + rank};
    const BPEMerge merge{merges_[rank]};
    if (merge.left >= output_value || merge.right >= output_value) {
      throw invalid_argument{"BPE merge parent does not precede its output token"};
    }
    const string& left{pieces_[merge.left]};
    const string& right{pieces_[merge.right]};
    if (left.size() > maximum_token_bytes || right.size() > maximum_token_bytes - left.size()) {
      throw invalid_argument{"BPE token expansion exceeds the 16 MiB safety limit"};
    }
    string piece;
    piece.reserve(left.size() + right.size());
    piece.append(left);
    piece.append(right);
    pieces_.push_back(std::move(piece));
    merge_ranks_.try_emplace(pair_key(merge.left, merge.right), rank);
  }
}

size_t ByteBPETokenizer::vocab_size() const noexcept {
  return pieces_.size();
}

size_t ByteBPETokenizer::merge_count() const noexcept {
  return merges_.size();
}

span<const BPEMerge> ByteBPETokenizer::merges() const noexcept {
  return merges_;
}

vector<TokenId> ByteBPETokenizer::encode(string_view bytes) const {
  if (bytes.empty()) {
    return {};
  }
  vector<Node> nodes;
  nodes.reserve(bytes.size());
  for (size_t index{0}; index < bytes.size(); ++index) {
    nodes.push_back({static_cast<TokenId>(static_cast<unsigned char>(bytes[index])),
                     index == 0 ? no_node : index - 1,
                     index + 1 == bytes.size() ? no_node : index + 1, true});
  }

  priority_queue<Candidate, vector<Candidate>, CandidateLater> candidates;
  const auto enqueue = [&](size_t left) {
    if (left == no_node || !nodes[left].alive || nodes[left].next == no_node) {
      return;
    }
    const size_t right{nodes[left].next};
    const auto found{merge_ranks_.find(pair_key(nodes[left].token, nodes[right].token))};
    if (found != merge_ranks_.end()) {
      candidates.push({found->second, left, right});
    }
  };
  for (size_t index{0}; index + 1 < nodes.size(); ++index) {
    enqueue(index);
  }

  while (!candidates.empty()) {
    const Candidate candidate{candidates.top()};
    candidates.pop();
    if (!nodes[candidate.left].alive || !nodes[candidate.right].alive ||
        nodes[candidate.left].next != candidate.right) {
      continue;
    }
    const auto current{
        merge_ranks_.find(pair_key(nodes[candidate.left].token, nodes[candidate.right].token))};
    if (current == merge_ranks_.end() || current->second != candidate.rank) {
      continue;
    }

    Node& left{nodes[candidate.left]};
    Node& right{nodes[candidate.right]};
    left.token = static_cast<TokenId>(base_vocab_size + candidate.rank);
    left.next = right.next;
    right.alive = false;
    if (right.next != no_node) {
      nodes[right.next].previous = candidate.left;
    }
    enqueue(left.previous);
    enqueue(candidate.left);
  }

  vector<TokenId> result;
  result.reserve(bytes.size());
  for (size_t index{0}; index != no_node; index = nodes[index].next) {
    result.push_back(nodes[index].token);
  }
  return result;
}

string ByteBPETokenizer::decode(span<const TokenId> tokens) const {
  size_t output_size{0};
  for (const TokenId token : tokens) {
    if (token >= pieces_.size()) {
      throw out_of_range{"Cannot decode a TokenId outside the BPE vocabulary"};
    }
    if (pieces_[token].size() > numeric_limits<size_t>::max() - output_size) {
      throw overflow_error{"Decoded byte string size overflow"};
    }
    output_size += pieces_[token].size();
  }
  string result;
  result.reserve(output_size);
  for (const TokenId token : tokens) {
    result.append(pieces_[token]);
  }
  return result;
}

ByteBPETokenizer train_byte_bpe(span<const string> documents, BPETrainingConfig config) {
  validate_training_config(config);
  vector<vector<TokenId>> corpus;
  corpus.reserve(documents.size());
  for (const string& document : documents) {
    corpus.push_back(byte_tokens(document));
  }

  vector<BPEMerge> merges;
  merges.reserve(min(config.target_vocab_size - base_vocab_size, static_cast<size_t>(4096)));
  while (base_vocab_size + merges.size() < config.target_vocab_size) {
    unordered_map<uint64_t, uint64_t> frequencies;
    for (const auto& document : corpus) {
      for (size_t index{0}; index + 1 < document.size(); ++index) {
        uint64_t& count{frequencies[pair_key(document[index], document[index + 1])]};
        if (count == numeric_limits<uint64_t>::max()) {
          throw overflow_error{"BPE pair frequency overflow"};
        }
        ++count;
      }
    }
    if (frequencies.empty()) {
      break;
    }

    uint64_t best_key{0};
    uint64_t best_frequency{0};
    for (const auto& [key, frequency] : frequencies) {
      if (frequency > best_frequency || (frequency == best_frequency && key < best_key)) {
        best_key = key;
        best_frequency = frequency;
      }
    }
    if (best_frequency < config.min_pair_frequency) {
      break;
    }

    const auto [left, right]{unpack_pair(best_key)};
    const TokenId output{static_cast<TokenId>(base_vocab_size + merges.size())};
    merges.push_back({left, right});
    for (auto& document : corpus) {
      vector<TokenId> replaced;
      replaced.reserve(document.size());
      for (size_t index{0}; index < document.size();) {
        if (index + 1 < document.size() && document[index] == left &&
            document[index + 1] == right) {
          replaced.push_back(output);
          index += 2;
        } else {
          replaced.push_back(document[index]);
          ++index;
        }
      }
      document = std::move(replaced);
    }
  }
  return ByteBPETokenizer{std::move(merges)};
}

} // namespace spar::tokenizer
