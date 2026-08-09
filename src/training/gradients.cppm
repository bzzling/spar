export module spar.training.gradients;

import std;
export import spar.nn.parameter;

export namespace spar::training {

/// Returns the stable global L2 norm over unique active Parameters. An active gradient belongs to
/// a Parameter that both requires gradients and currently has an accumulated gradient.
[[nodiscard]] double global_grad_norm(std::span<nn::Parameter> parameters);

/// Imperatively scales each unique active Parameter's existing gradient storage exactly once.
void scale_gradients(std::span<nn::Parameter> parameters, double factor);

struct ClipGradNormResult final {
  double total_norm;
  double scale;
  bool clipped;
};

/// Computes the global norm transactionally, then scales only when it exceeds `max_norm`.
/// Nonfinite norms throw before any gradient is modified.
[[nodiscard]] ClipGradNormResult clip_grad_norm(std::span<nn::Parameter> parameters,
                                                double max_norm);

} // namespace spar::training
