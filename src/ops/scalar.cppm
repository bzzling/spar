export module spar.ops.scalar;

export import spar.tensor;

export namespace spar {

[[nodiscard]] Tensor add_scalar(const Tensor& input, double value);
[[nodiscard]] Tensor subtract_scalar(const Tensor& input, double value);
[[nodiscard]] Tensor multiply_scalar(const Tensor& input, double value);
[[nodiscard]] Tensor divide_scalar(const Tensor& input, double value);

} // namespace spar
