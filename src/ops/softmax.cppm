export module spar.ops.softmax;

import std;
export import spar.tensor;

export namespace spar {

/// computes a numerically stable softmax over all tensor elements.
/// NaNs propagate; positive infinities share the probability mass equally;
/// an input containing only negative infinities produces all NaNs.
[[nodiscard]] Tensor softmax(const Tensor& input);
[[nodiscard]] Tensor softmax(const Tensor& input, std::size_t axis);

/// Computes stable log probabilities. Slices containing NaN, positive infinity,
/// or only negative infinities have deliberately undefined (NaN) derivatives.
[[nodiscard]] Tensor log_softmax(const Tensor& input);
[[nodiscard]] Tensor log_softmax(const Tensor& input, std::size_t axis);

} // namespace spar
