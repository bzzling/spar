export module spar.nn.rmsnorm;

import std;
export import spar.dtype;
export import spar.nn.parameter;
export import spar.tensor;

export namespace spar::nn {

class RMSNorm {
public:
  RMSNorm(std::size_t normalized_size, DType dtype, double epsilon = 1.0e-5);

  [[nodiscard]] Tensor forward(const Tensor& input) const;

  [[nodiscard]] std::size_t normalized_size() const noexcept;
  [[nodiscard]] double epsilon() const noexcept;
  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;

private:
  std::size_t normalized_size_;
  double epsilon_;
  Parameter weight_;
};

} // namespace spar::nn
