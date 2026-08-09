export module spar.leda.pretraining;

import std;
export import spar.checkpoint;
export import spar.data.batch_iterator;
export import spar.leda.model;
export import spar.optim.adamw;
export import spar.training;

export namespace spar::leda {

struct PretrainingConfig final {
  std::size_t sequence_length;
  std::size_t stride;
  std::size_t microbatch_size;
  std::size_t accumulation_steps;
  double max_grad_norm;
  training::WarmupCosineConfig learning_rate;
  double beta1{0.9};
  double beta2{0.95};
  double epsilon{1.0e-8};
  double weight_decay{0.1};
  std::uint64_t shuffle_seed;
};

void validate_pretraining_config(const PretrainingConfig& config);

struct TrainingStepResult final {
  double mean_loss;
  std::uint64_t target_count;
  std::size_t microbatches;
  double grad_norm;
  double clip_scale;
  bool clipped;
  double learning_rate;
};

/// Executes one Leda optimizer update. Exceptions are fatal for the current run: consumed batches,
/// tokens_seen, and partial gradients are not transactionally rolled back.
[[nodiscard]] std::optional<TrainingStepResult> train_update(Leda& model, optim::AdamW& optimizer,
                                                             data::LMBatchIterator& batches,
                                                             checkpoint::TrainingProgress& progress,
                                                             const PretrainingConfig& config);

struct EvaluationResult final {
  double mean_loss;
  std::uint64_t target_count;
  std::size_t batches;
};

/// Computes one globally target-weighted validation pass without calling backward. Temporary
/// forward graphs are still constructed because Spar has no no-grad mode yet.
[[nodiscard]] EvaluationResult evaluate(const Leda& model,
                                        data::LMBatchIterator& validation_batches);

} // namespace spar::leda
