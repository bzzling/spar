export module spar.nn.parameter;

export import spar.tensor;

export namespace spar::nn {

/// A trainable floating-point leaf Tensor. Copies share value/gradient identity;
/// clone() creates an independent leaf.
class Parameter {
public:
  explicit Parameter(Tensor initial_value);

  [[nodiscard]] Tensor& tensor() noexcept;
  [[nodiscard]] const Tensor& tensor() const noexcept;

  [[nodiscard]] bool requires_grad() const noexcept;
  void set_requires_grad(bool enabled);
  [[nodiscard]] bool has_grad() const noexcept;
  [[nodiscard]] Tensor grad() const;
  void zero_grad();

  /// Returns an independent trainable leaf with copied values and no gradient.
  [[nodiscard]] Parameter clone() const;

private:
  Tensor tensor_;
};

} // namespace spar::nn
