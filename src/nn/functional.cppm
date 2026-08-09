export module spar.nn.functional;

export import spar.tensor;

export namespace spar::nn {

[[nodiscard]] Tensor swiglu(const Tensor& gate, const Tensor& value);

} // namespace spar::nn
