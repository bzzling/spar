module spar.training.schedule;

import std;

using namespace std;

namespace spar::training {

double warmup_cosine_learning_rate(WarmupCosineConfig config, uint64_t completed_steps) {
  if (!isfinite(config.peak_learning_rate) || config.peak_learning_rate < 0.0) {
    throw invalid_argument{"Peak learning rate must be finite and nonnegative"};
  }
  if (!isfinite(config.minimum_learning_rate) || config.minimum_learning_rate < 0.0) {
    throw invalid_argument{"Minimum learning rate must be finite and nonnegative"};
  }
  if (config.minimum_learning_rate > config.peak_learning_rate) {
    throw invalid_argument{"Minimum learning rate must not exceed peak learning rate"};
  }
  if (config.decay_steps == 0) {
    throw invalid_argument{"Cosine decay must contain at least one update"};
  }

  if (completed_steps < config.warmup_steps) {
    const double numerator{static_cast<double>(completed_steps + 1U)};
    const double progress{numerator / static_cast<double>(config.warmup_steps)};
    return config.peak_learning_rate * progress;
  }

  const uint64_t decay_index{completed_steps - config.warmup_steps};
  if (decay_index >= config.decay_steps || config.decay_steps == 1) {
    return config.minimum_learning_rate;
  }
  const double progress{static_cast<double>(decay_index) /
                        static_cast<double>(config.decay_steps - 1U)};
  const double cosine{cos(numbers::pi_v<double> * progress)};
  return config.minimum_learning_rate +
         0.5 * (config.peak_learning_rate - config.minimum_learning_rate) * (1.0 + cosine);
}

} // namespace spar::training
