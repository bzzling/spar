module spar.data.tokenized_corpus;

import std;
import spar.tokenizer.byte_bpe;

using namespace std;

namespace spar::data {

TokenizedCorpus::TokenizedCorpus(span<const string> documents,
                                 const tokenizer::ByteBPETokenizer& tokenizer)
    : document_count_{documents.size()}, eod_token_id_{0} {
  if (tokenizer.vocab_size() > numeric_limits<tokenizer::TokenId>::max() ||
      tokenizer.vocab_size() == numeric_limits<size_t>::max()) {
    throw invalid_argument{"Tokenizer vocabulary leaves no representable model-level EOD token"};
  }
  eod_token_id_ = static_cast<tokenizer::TokenId>(tokenizer.vocab_size());
  for (const string& document : documents) {
    auto encoded{tokenizer.encode(document)};
    const size_t available{numeric_limits<size_t>::max() - tokens_.size()};
    if (encoded.size() >= available) {
      throw overflow_error{"Packed token corpus size overflow"};
    }
    tokens_.insert(tokens_.end(), encoded.begin(), encoded.end());
    tokens_.push_back(eod_token_id_);
  }
}

size_t TokenizedCorpus::document_count() const noexcept {
  return document_count_;
}

size_t TokenizedCorpus::token_count() const noexcept {
  return tokens_.size();
}

tokenizer::TokenId TokenizedCorpus::eod_token_id() const noexcept {
  return eod_token_id_;
}

size_t TokenizedCorpus::model_vocab_size() const noexcept {
  return static_cast<size_t>(eod_token_id_) + 1;
}

span<const tokenizer::TokenId> TokenizedCorpus::tokens() const noexcept {
  return tokens_;
}

WindowDataset::WindowDataset(const TokenizedCorpus& corpus, WindowConfig config)
    : corpus_{&corpus}, config_{config}, window_count_{0} {
  if (config_.sequence_length < 2) {
    throw invalid_argument{"LM window sequence length must be at least two"};
  }
  if (config_.stride == 0 || config_.stride > config_.sequence_length) {
    throw invalid_argument{"LM window stride must be in [1, sequence_length]"};
  }
  if (corpus.token_count() >= config_.sequence_length) {
    window_count_ = 1 + (corpus.token_count() - config_.sequence_length) / config_.stride;
  }
}

size_t WindowDataset::window_count() const noexcept {
  return window_count_;
}

size_t WindowDataset::sequence_length() const noexcept {
  return config_.sequence_length;
}

size_t WindowDataset::stride() const noexcept {
  return config_.stride;
}

span<const tokenizer::TokenId> WindowDataset::window(size_t index) const {
  if (index >= window_count_) {
    throw out_of_range{"LM window index is out of range"};
  }
  const size_t start{index * config_.stride};
  return corpus_->tokens().subspan(start, config_.sequence_length);
}

const TokenizedCorpus& WindowDataset::corpus() const noexcept {
  return *corpus_;
}

} // namespace spar::data
