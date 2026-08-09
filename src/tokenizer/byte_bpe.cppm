export module spar.tokenizer.byte_bpe;

import std;

export namespace spar::tokenizer {

using TokenId = std::uint32_t;

struct BPEMerge final {
  TokenId left;
  TokenId right;

  friend bool operator==(const BPEMerge&, const BPEMerge&) = default;
};

struct BPETrainingConfig final {
  std::size_t target_vocab_size{256};
  std::size_t min_pair_frequency{2};
};

/// A lossless raw-byte BPE tokenizer. IDs 0..255 are the corresponding literal bytes; learned
/// token 256+r is produced by merge rank r. No Unicode normalization or pretokenization occurs.
/// This is intentionally not GPT-2, Hugging Face, SentencePiece, or tiktoken compatibility: bytes
/// are used directly, without regex splitting or a byte-to-Unicode remapping. There are no special
/// or unknown tokens: the fixed byte vocabulary makes every input representable.
class ByteBPETokenizer final {
public:
  ByteBPETokenizer();
  explicit ByteBPETokenizer(std::vector<BPEMerge> merges);

  [[nodiscard]] std::size_t vocab_size() const noexcept;
  [[nodiscard]] std::size_t merge_count() const noexcept;
  [[nodiscard]] std::span<const BPEMerge> merges() const noexcept;

  [[nodiscard]] std::vector<TokenId> encode(std::string_view bytes) const;
  /// Decodes any valid token stream. Re-encoding its bytes produces the canonical BPE form, which
  /// need not equal a caller-supplied noncanonical token sequence.
  [[nodiscard]] std::string decode(std::span<const TokenId> tokens) const;

private:
  std::vector<BPEMerge> merges_;
  std::vector<std::string> pieces_;
  std::unordered_map<std::uint64_t, std::size_t> merge_ranks_;
};

/// Trains with full pair recounts. Documents are hard boundaries and never contribute cross-file
/// pairs. Equal frequencies choose the lexicographically smallest (left,right) token-ID pair.
[[nodiscard]] ByteBPETokenizer train_byte_bpe(std::span<const std::string> documents,
                                              BPETrainingConfig config);

} // namespace spar::tokenizer
