export module spar.nn.mlp;

import std;
export import spar.dtype;
export import spar.nn.linear;
export import spar.random;
export import spar.tensor;

export namespace spar::nn {

class SwiGLUMLP {
public:
  SwiGLUMLP(std::size_t model_dim, std::size_t hidden_dim, DType dtype, Random& random,
            bool bias = false);

  [[nodiscard]] Tensor forward(const Tensor& input) const;

  [[nodiscard]] std::size_t model_dim() const noexcept;
  [[nodiscard]] std::size_t hidden_dim() const noexcept;

  [[nodiscard]] Linear& gate_proj() noexcept;
  [[nodiscard]] const Linear& gate_proj() const noexcept;
  [[nodiscard]] Linear& up_proj() noexcept;
  [[nodiscard]] const Linear& up_proj() const noexcept;
  [[nodiscard]] Linear& down_proj() noexcept;
  [[nodiscard]] const Linear& down_proj() const noexcept;

private:
  std::size_t model_dim_;
  std::size_t hidden_dim_;
  Linear gate_proj_;
  Linear up_proj_;
  Linear down_proj_;
};

} // namespace spar::nn
