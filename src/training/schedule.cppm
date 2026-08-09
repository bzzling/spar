export module spar.training.schedule;

import std;

export namespace spar::training {

struct WarmupCosineConfig final {
  double peak_learning_rate;
  double minimum_learning_rate;
  std::uint64_t warmup_steps;
  std::uint64_t decay_steps;
};

/// `completed_steps` is the number of optimizer updates completed before the update being
/// scheduled. The schedule has no evolving state and is reconstructed from this value alone.
[[nodiscard]] double warmup_cosine_learning_rate(WarmupCosineConfig config,
                                                 std::uint64_t completed_steps);

} // namespace spar::training
