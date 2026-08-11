export module spar.nn.parameter;

import std;
export import spar.tensor;

export namespace spar::nn {

/// A trainable floating-point leaf Tensor. Copies share value/gradient identity;
/// clone() creates an independent leaf.
class Parameter {
public:
  explicit Parameter(Tensor initial_value);

  [[nodiscard]] const Tensor& tensor() const noexcept;

  [[nodiscard]] bool requires_grad() const noexcept;
  [[nodiscard]] bool shares_identity_with(const Parameter& other) const noexcept;
  void set_requires_grad(bool enabled);
  [[nodiscard]] bool has_grad() const noexcept;
  [[nodiscard]] Tensor grad() const;
  void zero_grad();

  /// Returns an independent trainable leaf with copied values and no gradient.
  [[nodiscard]] Parameter clone() const;

private:
  friend void move_to(Parameter& parameter, Device target);
  friend void move_to(std::span<Parameter> parameters, Device target);
  Tensor tensor_;
};

/// Mutates placement while preserving the Parameter's Tensor, autograd, and Storage identities.
/// This is a graph-boundary operation and rejects migration when a live gradient exists.
void move_to(Parameter& parameter, Device target);

/// Transactionally migrates each unique Parameter while preserving aliases and ties.
/// All allocations and copies complete before the no-throw commit phase begins.
void move_to(std::span<Parameter> parameters, Device target);

} // namespace spar::nn
