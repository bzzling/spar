module spar.loss.cross_entropy;

import std;
import spar.dtype;
import spar.ops.reduction;
import spar.ops.scalar;
import spar.ops.softmax;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar::loss {
namespace {

vector<size_t> validated_targets(const Tensor& targets, size_t classes) {
  if (targets.dtype() != DType::Int32 && targets.dtype() != DType::Int64) {
    throw invalid_argument{"cross_entropy targets must have Int32 or Int64 dtype"};
  }
  if (targets.requires_grad()) {
    throw invalid_argument{"cross_entropy targets must not require gradients"};
  }
  vector<size_t> result;
  result.reserve(targets.numel());
  const auto append = [&]<typename T> {
    for (size_t index{0}; index < targets.numel(); ++index) {
      const T target{detail::logical_value<T>(targets, index)};
      if (target < 0 || static_cast<uint64_t>(target) >= classes) {
        throw out_of_range{"cross_entropy target is outside the class range"};
      }
      result.push_back(static_cast<size_t>(target));
    }
  };
  if (targets.dtype() == DType::Int32) {
    append.template operator()<int32_t>();
  } else {
    append.template operator()<int64_t>();
  }
  return result;
}

template <typename T>
Tensor nll_values(const Tensor& log_probabilities, span<const size_t> targets,
                  span<const size_t> source_rows, Shape output_shape) {
  const size_t classes{
      static_cast<size_t>(log_probabilities.shape()[log_probabilities.rank() - 1])};
  Tensor output{std::move(output_shape), log_probabilities.dtype()};
  auto losses{output.span<T>()};
  for (size_t index{0}; index < targets.size(); ++index) {
    losses[index] =
        -detail::logical_value<T>(log_probabilities, source_rows[index] * classes + targets[index]);
  }
  return output;
}

template <typename T>
Tensor nll_gradient(const Tensor& gradient, const Shape& log_probability_shape, DType dtype,
                    span<const size_t> targets, span<const size_t> source_rows) {
  const size_t classes{
      static_cast<size_t>(log_probability_shape[log_probability_shape.rank() - 1])};
  Tensor contribution{zeros(log_probability_shape, dtype)};
  auto values{contribution.span<T>()};
  const auto upstream{gradient.span<T>()};
  for (size_t index{0}; index < targets.size(); ++index) {
    values[source_rows[index] * classes + targets[index]] -= upstream[index];
  }
  return contribution;
}

Tensor select_nll(const Tensor& log_probabilities, vector<size_t> targets,
                  vector<size_t> source_rows, Shape output_shape) {
  Tensor output{log_probabilities.dtype() == DType::Float32
                    ? nll_values<float>(log_probabilities, targets, source_rows, output_shape)
                    : nll_values<double>(log_probabilities, targets, source_rows, output_shape)};
  if (log_probabilities.requires_grad()) {
    const Shape input_shape{log_probabilities.shape()};
    const DType dtype{log_probabilities.dtype()};
    detail::record_operation(output, {log_probabilities},
                             [input_shape, dtype, targets = std::move(targets),
                              source_rows = std::move(source_rows)](const Tensor& gradient) {
                               if (dtype == DType::Float32) {
                                 return vector<Tensor>{nll_gradient<float>(
                                     gradient, input_shape, dtype, targets, source_rows)};
                               }
                               return vector<Tensor>{nll_gradient<double>(
                                   gradient, input_shape, dtype, targets, source_rows)};
                             });
  }
  return output;
}

Tensor reduce_loss(const Tensor& losses, Reduction reduction) {
  switch (reduction) {
  case Reduction::None:
    return losses;
  case Reduction::Sum:
    return sum(losses);
  case Reduction::Mean:
    return mean(losses);
  }
  throw invalid_argument{"cross_entropy received an unknown Reduction"};
}

void validate_logits(const Tensor& logits) {
  if (logits.dtype() != DType::Float32 && logits.dtype() != DType::Float64) {
    throw invalid_argument{"cross_entropy logits must have Float32 or Float64 dtype"};
  }
  if (logits.rank() == 0) {
    throw invalid_argument{"cross_entropy logits must have a class dimension"};
  }
  if (logits.shape()[logits.rank() - 1] <= 0) {
    throw invalid_argument{"cross_entropy class dimension must be positive"};
  }
}

} // namespace

Tensor cross_entropy(const Tensor& logits, const Tensor& targets, Reduction reduction) {
  validate_logits(logits);
  if (targets.numel() == 0) {
    throw invalid_argument{"cross_entropy requires at least one target"};
  }
  vector<Shape::dimension_type> expected_dimensions{logits.shape().dimensions().begin(),
                                                    logits.shape().dimensions().end() - 1};
  if (targets.shape() != Shape{expected_dimensions}) {
    throw invalid_argument{"cross_entropy target shape does not match logits"};
  }
  const size_t classes{static_cast<size_t>(logits.shape()[logits.rank() - 1])};
  vector<size_t> targets_snapshot{validated_targets(targets, classes)};
  vector<size_t> rows(targets.numel());
  for (size_t index{0}; index < rows.size(); ++index) {
    rows[index] = index;
  }
  return reduce_loss(select_nll(log_softmax(logits, logits.rank() - 1), std::move(targets_snapshot),
                                std::move(rows), targets.shape()),
                     reduction);
}

Tensor language_model_cross_entropy(const Tensor& logits, const Tensor& token_ids,
                                    Reduction reduction) {
  validate_logits(logits);
  if (logits.rank() != 3 || token_ids.rank() != 2) {
    throw invalid_argument{"language_model_cross_entropy requires logits [B,T,V] and IDs [B,T]"};
  }
  if (logits.shape()[0] <= 0 || logits.shape()[1] < 2 ||
      token_ids.shape()[0] != logits.shape()[0] || token_ids.shape()[1] != logits.shape()[1]) {
    throw invalid_argument{"language_model_cross_entropy received incompatible shapes"};
  }
  const size_t batch{static_cast<size_t>(logits.shape()[0])};
  const size_t sequence{static_cast<size_t>(logits.shape()[1])};
  const size_t classes{static_cast<size_t>(logits.shape()[2])};
  const vector<size_t> all_tokens{validated_targets(token_ids, classes)};
  vector<size_t> targets;
  vector<size_t> rows;
  targets.reserve(batch * (sequence - 1));
  rows.reserve(batch * (sequence - 1));
  for (size_t batch_index{0}; batch_index < batch; ++batch_index) {
    for (size_t time{0}; time + 1 < sequence; ++time) {
      targets.push_back(all_tokens[batch_index * sequence + time + 1]);
      rows.push_back(batch_index * sequence + time);
    }
  }
  const auto dimension = [](size_t value) { return static_cast<Shape::dimension_type>(value); };
  return reduce_loss(select_nll(log_softmax(logits, 2), std::move(targets), std::move(rows),
                                Shape{dimension(batch), dimension(sequence - 1)}),
                     reduction);
}

} // namespace spar::loss
