export module spar.ops.softmax;

export import spar.tensor;

export namespace spar {

/// computes a numerically stable softmax over all tensor elements.
/// NaNs propagate; positive infinities share the probability mass equally;
/// an input containing only negative infinities produces all NaNs.
[[nodiscard]] Tensor softmax(const Tensor& input);

} // namespace spar
