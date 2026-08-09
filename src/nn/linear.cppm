export module spar.nn.linear;

import std;
export import spar.dtype;
export import spar.nn.parameter;
export import spar.random;
export import spar.tensor;

export namespace spar::nn {

class Linear {
public:
  Linear(std::size_t in_features, std::size_t out_features, DType dtype, Random& random,
         bool bias = true);

  [[nodiscard]] Tensor forward(const Tensor& input) const;

  [[nodiscard]] std::size_t in_features() const noexcept;
  [[nodiscard]] std::size_t out_features() const noexcept;
  [[nodiscard]] bool has_bias() const noexcept;
  [[nodiscard]] Parameter& weight() noexcept;
  [[nodiscard]] const Parameter& weight() const noexcept;
  [[nodiscard]] Parameter* bias() noexcept;
  [[nodiscard]] const Parameter* bias() const noexcept;

private:
  std::size_t in_features_;
  std::size_t out_features_;
  Parameter weight_;
  std::optional<Parameter> bias_;
};

} // namespace spar::nn
