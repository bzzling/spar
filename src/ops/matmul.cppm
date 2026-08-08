export module spar.ops.matmul;

export import spar.tensor;

export namespace spar {

[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace spar
