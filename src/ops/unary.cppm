export module spar.ops.unary;

export import spar.tensor;

export namespace spar {

[[nodiscard]] Tensor negate(const Tensor& input);
[[nodiscard]] Tensor square(const Tensor& input);
[[nodiscard]] Tensor reciprocal(const Tensor& input);
[[nodiscard]] Tensor exp(const Tensor& input);
[[nodiscard]] Tensor log(const Tensor& input);
[[nodiscard]] Tensor sqrt(const Tensor& input);
/// uses separate nonnegative and negative branches to avoid unnecessary overflow.
[[nodiscard]] Tensor sigmoid(const Tensor& input);
[[nodiscard]] Tensor silu(const Tensor& input);

} // namespace spar
