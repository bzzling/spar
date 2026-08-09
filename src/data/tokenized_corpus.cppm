export module spar.data.tokenized_corpus;

import std;
export import spar.tokenizer.byte_bpe;

export namespace spar::data {

/// Owns one packed token stream: encode(doc0), EOD, encode(doc1), EOD, ... . Empty and final
/// documents also receive exactly one EOD. EOD is a model-level token, not a tokenizer token.
class TokenizedCorpus final {
public:
  TokenizedCorpus(std::span<const std::string> documents,
                  const tokenizer::ByteBPETokenizer& tokenizer);

  [[nodiscard]] std::size_t document_count() const noexcept;
  [[nodiscard]] std::size_t token_count() const noexcept;
  [[nodiscard]] tokenizer::TokenId eod_token_id() const noexcept;
  [[nodiscard]] std::size_t model_vocab_size() const noexcept;
  [[nodiscard]] std::span<const tokenizer::TokenId> tokens() const noexcept;

private:
  std::size_t document_count_;
  tokenizer::TokenId eod_token_id_;
  std::vector<tokenizer::TokenId> tokens_;
};

struct WindowConfig final {
  std::size_t sequence_length;
  std::size_t stride;
};

/// Borrows `corpus`; the corpus must outlive this dataset and every iterator borrowing it. Windows
/// are arithmetic spans over the single packed stream, may cross EOD, and never pad or wrap.
class WindowDataset final {
public:
  WindowDataset(const TokenizedCorpus& corpus, WindowConfig config);

  [[nodiscard]] std::size_t window_count() const noexcept;
  [[nodiscard]] std::size_t sequence_length() const noexcept;
  [[nodiscard]] std::size_t stride() const noexcept;
  [[nodiscard]] std::span<const tokenizer::TokenId> window(std::size_t index) const;
  [[nodiscard]] const TokenizedCorpus& corpus() const noexcept;

private:
  const TokenizedCorpus* corpus_;
  WindowConfig config_;
  std::size_t window_count_;
};

} // namespace spar::data
