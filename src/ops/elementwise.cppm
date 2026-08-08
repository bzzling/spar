export module spar.ops.elementwise;

export import spar.tensor;

export namespace spar {

[[nodiscard]] Tensor add(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor subtract(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor multiply(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor divide(const Tensor& a, const Tensor& b);

} // namespace spar
