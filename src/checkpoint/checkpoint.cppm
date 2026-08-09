export module spar.checkpoint;

import std;
export import spar.nn.decoder;
export import spar.nn.state;
export import spar.optim.adamw;
export import spar.random;

export namespace spar::checkpoint {

struct TrainingProgress final {
  std::uint64_t global_step{0};
  std::uint64_t tokens_seen{0};
};

struct LoadedTrainingCheckpoint final {
  nn::DecoderLM model;
  optim::AdamW optimizer;
  Random random;
  TrainingProgress progress;
};

void save_training_checkpoint(const std::filesystem::path& path, nn::DecoderLM& model,
                              const optim::AdamW& optimizer, const Random& random,
                              TrainingProgress progress);

[[nodiscard]] LoadedTrainingCheckpoint load_training_checkpoint(const std::filesystem::path& path);

} // namespace spar::checkpoint
