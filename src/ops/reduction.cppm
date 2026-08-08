export module spar.ops.reduction;

export import spar.tensor;

export namespace spar {

[[nodiscard]] Tensor sum(const Tensor& input);
[[nodiscard]] Tensor mean(const Tensor& input);
[[nodiscard]] Tensor reduce_max(const Tensor& input);

} // namespace spar
