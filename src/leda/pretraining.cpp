module spar.leda.pretraining;

import std;
import spar.checkpoint;
import spar.data.batch_iterator;
import spar.leda.model;
import spar.loss.cross_entropy;
import spar.nn.parameter;
import spar.optim.adamw;
import spar.training.gradients;
import spar.training.schedule;

using namespace std;

namespace spar::leda {
namespace {

uint64_t target_count(const Tensor& batch) {
  if (batch.rank() != 2 || batch.shape()[0] <= 0 || batch.shape()[1] < 2) {
    throw invalid_argument{"Leda pretraining batch must have shape [B,T] with B > 0 and T >= 2"};
  }
  const auto rows{static_cast<uint64_t>(batch.shape()[0])};
  const auto targets_per_row{static_cast<uint64_t>(batch.shape()[1] - 1)};
  if (rows > numeric_limits<uint64_t>::max() / targets_per_row) {
    throw overflow_error{"Leda target count overflow"};
  }
  return rows * targets_per_row;
}

void checked_add(uint64_t& destination, uint64_t value, string_view description) {
  if (value > numeric_limits<uint64_t>::max() - destination) {
    throw overflow_error{string{"Leda "} + string{description} + " overflow"};
  }
  destination += value;
}

double scalar_value(const Tensor& tensor) {
  if (tensor.numel() != 1 ||
      (tensor.dtype() != DType::Float32 && tensor.dtype() != DType::Float64)) {
    throw logic_error{"Leda loss must be one floating-point scalar"};
  }
  return tensor.dtype() == DType::Float32 ? static_cast<double>(tensor.span<float>()[0])
                                          : tensor.span<double>()[0];
}

} // namespace

void validate_pretraining_config(const PretrainingConfig& config) {
  if (config.sequence_length < 2) {
    throw invalid_argument{"Leda pretraining sequence length must be at least two"};
  }
  if (config.stride == 0 || config.stride > config.sequence_length) {
    throw invalid_argument{"Leda pretraining stride must be in [1, sequence_length]"};
  }
  if (config.microbatch_size == 0 || config.accumulation_steps == 0) {
    throw invalid_argument{"Leda microbatch size and accumulation steps must be positive"};
  }
  if (!isfinite(config.max_grad_norm) || config.max_grad_norm < 0.0) {
    throw invalid_argument{"Leda maximum gradient norm must be finite and nonnegative"};
  }
  static_cast<void>(training::warmup_cosine_learning_rate(config.learning_rate, 0));
  static_cast<void>(optim::AdamW{{},
                                 config.learning_rate.peak_learning_rate,
                                 config.beta1,
                                 config.beta2,
                                 config.epsilon,
                                 config.weight_decay});
}

optional<TrainingStepResult> train_update(Leda& model, optim::AdamW& optimizer,
                                          data::LMBatchIterator& batches,
                                          checkpoint::TrainingProgress& progress,
                                          const PretrainingConfig& config) {
  validate_pretraining_config(config);
  auto model_parameters{parameters(model)};
  if (!ranges::all_of(model_parameters, [&optimizer](const nn::Parameter& parameter) {
        return optimizer.tracks(parameter);
      })) {
    throw invalid_argument{"AdamW does not track every Leda Parameter"};
  }
  if (progress.global_step == numeric_limits<uint64_t>::max()) {
    throw overflow_error{"Leda global_step overflow"};
  }
  optimizer.zero_grad();
  double summed_loss{0.0};
  uint64_t targets{0};
  size_t microbatches{0};
  for (; microbatches < config.accumulation_steps; ++microbatches) {
    optional<Tensor> batch{batches.next_batch()};
    if (!batch) {
      break;
    }
    if (static_cast<size_t>(batch->shape()[0]) > config.microbatch_size ||
        static_cast<size_t>(batch->shape()[1]) != config.sequence_length) {
      throw invalid_argument{"Leda batch shape does not match PretrainingConfig"};
    }
    const uint64_t batch_targets{target_count(*batch)};
    Tensor loss{
        loss::language_model_cross_entropy(model.forward(*batch), *batch, loss::Reduction::Sum)};
    summed_loss += scalar_value(loss);
    loss.backward();
    checked_add(targets, batch_targets, "accumulation target count");
    checked_add(progress.tokens_seen, batch_targets, "tokens_seen");
  }
  if (microbatches == 0) {
    return nullopt;
  }
  training::scale_gradients(model_parameters, 1.0 / static_cast<double>(targets));
  const training::ClipGradNormResult clip{
      training::clip_grad_norm(model_parameters, config.max_grad_norm)};
  const double learning_rate{
      training::warmup_cosine_learning_rate(config.learning_rate, progress.global_step)};
  optimizer.set_learning_rate(learning_rate);
  optimizer.step();
  ++progress.global_step;
  return TrainingStepResult{.mean_loss = summed_loss / static_cast<double>(targets),
                            .target_count = targets,
                            .microbatches = microbatches,
                            .grad_norm = clip.total_norm,
                            .clip_scale = clip.scale,
                            .clipped = clip.clipped,
                            .learning_rate = learning_rate};
}

EvaluationResult evaluate(const Leda& model, data::LMBatchIterator& validation_batches) {
  double summed_loss{0.0};
  uint64_t targets{0};
  size_t batch_count{0};
  while (optional<Tensor> batch{validation_batches.next_batch()}) {
    const uint64_t batch_targets{target_count(*batch)};
    const Tensor loss{
        loss::language_model_cross_entropy(model.forward(*batch), *batch, loss::Reduction::Sum)};
    summed_loss += scalar_value(loss);
    checked_add(targets, batch_targets, "evaluation target count");
    ++batch_count;
  }
  if (targets == 0) {
    throw invalid_argument{"Leda evaluation iterator contains no targets"};
  }
  return {.mean_loss = summed_loss / static_cast<double>(targets),
          .target_count = targets,
          .batches = batch_count};
}

} // namespace spar::leda
