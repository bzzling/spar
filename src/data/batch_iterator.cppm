export module spar.data.batch_iterator;

import std;
export import spar.dtype;
export import spar.tensor;
export import spar.data.tokenized_corpus;

export namespace spar::data {

struct BatchConfig final {
  std::size_t batch_size;
  DType token_dtype{DType::Int32};
  bool shuffle{true};
  std::uint64_t shuffle_seed{0};
  bool drop_last{false};
};

struct BatchIteratorState final {
  std::uint64_t epoch{0};
  std::uint64_t cursor{0};

  friend bool operator==(const BatchIteratorState&, const BatchIteratorState&) = default;
};

/// Borrows `dataset`, which must outlive the iterator. State is exactly the epoch and number of
/// epoch-order windows already consumed; permutations are reconstructed from config and epoch.
class LMBatchIterator final {
public:
  LMBatchIterator(const WindowDataset& dataset, BatchConfig config);

  [[nodiscard]] std::optional<Tensor> next_batch();
  [[nodiscard]] BatchIteratorState state() const noexcept;
  void set_state(BatchIteratorState state);
  void next_epoch();
  [[nodiscard]] std::uint64_t epoch() const noexcept;
  [[nodiscard]] std::size_t remaining_windows() const noexcept;

private:
  void rebuild_order();

  const WindowDataset* dataset_;
  BatchConfig config_;
  std::vector<std::size_t> order_;
  BatchIteratorState state_;
};

struct DocumentSplit final {
  std::vector<std::size_t> train_indices;
  std::vector<std::size_t> validation_indices;

  friend bool operator==(const DocumentSplit&, const DocumentSplit&) = default;
};

/// Deterministically shuffles document indices, returning the first `validation_count` for
/// validation and the remainder for training, preserving shuffled order within both results.
[[nodiscard]] DocumentSplit split_documents(std::size_t document_count,
                                            std::size_t validation_count, std::uint64_t seed);

} // namespace spar::data
