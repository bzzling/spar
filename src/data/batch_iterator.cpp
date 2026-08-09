module spar.data.batch_iterator;

import std;
import spar.data.tokenized_corpus;
import spar.dtype;
import spar.random;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar::data {
namespace {

uint64_t derive_epoch_seed(uint64_t seed, uint64_t epoch) noexcept {
  uint64_t value{seed + 0x9E3779B97F4A7C15ULL * (epoch + 1U)};
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

uint64_t bounded_index(Random& random, uint64_t bound) {
  if (bound == 0) {
    throw logic_error{"Internal bounded sampling requires a positive bound"};
  }
  const uint64_t threshold{(uint64_t{0} - bound) % bound};
  uint64_t value{0};
  do {
    value = random.next_u64();
  } while (value < threshold);
  return value % bound;
}

void fisher_yates(vector<size_t>& indices, uint64_t seed) {
  Random random{seed};
  for (size_t remaining{indices.size()}; remaining > 1; --remaining) {
    const auto chosen{static_cast<size_t>(bounded_index(random, static_cast<uint64_t>(remaining)))};
    std::swap(indices[remaining - 1], indices[chosen]);
  }
}

template <typename T>
void copy_batch_rows(Tensor& batch, const WindowDataset& dataset, span<const size_t> indices) {
  auto output{batch.span<T>()};
  const size_t sequence_length{dataset.sequence_length()};
  for (size_t row{0}; row < indices.size(); ++row) {
    const auto source{dataset.window(indices[row])};
    for (size_t column{0}; column < sequence_length; ++column) {
      output[row * sequence_length + column] = static_cast<T>(source[column]);
    }
  }
}

} // namespace

LMBatchIterator::LMBatchIterator(const WindowDataset& dataset, BatchConfig config)
    : dataset_{&dataset}, config_{config}, state_{} {
  if (config_.batch_size == 0) {
    throw invalid_argument{"LM batch size must be positive"};
  }
  if (config_.token_dtype != DType::Int32 && config_.token_dtype != DType::Int64) {
    throw invalid_argument{"LM batches require int32 or int64 token dtype"};
  }
  if (config_.batch_size > static_cast<size_t>(numeric_limits<int64_t>::max()) ||
      dataset.sequence_length() > static_cast<size_t>(numeric_limits<int64_t>::max())) {
    throw invalid_argument{"LM batch shape exceeds Spar Shape dimension range"};
  }
  if (dataset.window_count() > numeric_limits<uint64_t>::max()) {
    throw invalid_argument{"LM window count exceeds iterator-state range"};
  }
  if (config_.token_dtype == DType::Int32 &&
      dataset.corpus().eod_token_id() >
          static_cast<tokenizer::TokenId>(numeric_limits<int32_t>::max())) {
    throw invalid_argument{"LM corpus token IDs do not fit int32"};
  }
  rebuild_order();
}

optional<Tensor> LMBatchIterator::next_batch() {
  const size_t cursor{static_cast<size_t>(state_.cursor)};
  if (cursor == order_.size()) {
    return nullopt;
  }
  const size_t remaining{order_.size() - cursor};
  if (config_.drop_last && remaining < config_.batch_size) {
    state_.cursor = static_cast<uint64_t>(order_.size());
    return nullopt;
  }
  const size_t rows{min(remaining, config_.batch_size)};
  Tensor batch{Shape{static_cast<int64_t>(rows), static_cast<int64_t>(dataset_->sequence_length())},
               config_.token_dtype};
  const span<const size_t> indices{order_.data() + cursor, rows};
  switch (config_.token_dtype) {
  case DType::Int32:
    copy_batch_rows<int32_t>(batch, *dataset_, indices);
    break;
  case DType::Int64:
    copy_batch_rows<int64_t>(batch, *dataset_, indices);
    break;
  case DType::Float32:
  case DType::Float64:
    throw logic_error{"Invalid internal LM batch dtype"};
  }
  state_.cursor += static_cast<uint64_t>(rows);
  return batch;
}

BatchIteratorState LMBatchIterator::state() const noexcept {
  return state_;
}

void LMBatchIterator::set_state(BatchIteratorState state) {
  if (state.cursor > static_cast<uint64_t>(dataset_->window_count())) {
    throw invalid_argument{"LM batch iterator cursor exceeds the window count"};
  }
  state_ = state;
  rebuild_order();
}

void LMBatchIterator::next_epoch() {
  if (state_.epoch == numeric_limits<uint64_t>::max()) {
    throw overflow_error{"LM batch iterator epoch overflow"};
  }
  ++state_.epoch;
  state_.cursor = 0;
  rebuild_order();
}

uint64_t LMBatchIterator::epoch() const noexcept {
  return state_.epoch;
}

size_t LMBatchIterator::remaining_windows() const noexcept {
  return order_.size() - static_cast<size_t>(state_.cursor);
}

void LMBatchIterator::rebuild_order() {
  order_.resize(dataset_->window_count());
  std::iota(order_.begin(), order_.end(), size_t{0});
  if (config_.shuffle) {
    fisher_yates(order_, derive_epoch_seed(config_.shuffle_seed, state_.epoch));
  }
}

DocumentSplit split_documents(size_t document_count, size_t validation_count, uint64_t seed) {
  if (validation_count > document_count) {
    throw invalid_argument{"Validation document count exceeds total document count"};
  }
  if (document_count > numeric_limits<uint64_t>::max() ||
      document_count > static_cast<size_t>(numeric_limits<ptrdiff_t>::max())) {
    throw invalid_argument{"Document count exceeds supported iterator range"};
  }
  vector<size_t> indices(document_count);
  std::iota(indices.begin(), indices.end(), size_t{0});
  fisher_yates(indices, seed);
  DocumentSplit result;
  result.validation_indices.assign(indices.begin(),
                                   indices.begin() + static_cast<ptrdiff_t>(validation_count));
  result.train_indices.assign(indices.begin() + static_cast<ptrdiff_t>(validation_count),
                              indices.end());
  return result;
}

} // namespace spar::data
