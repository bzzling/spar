export module spar.optim.adamw;

import std;
export import spar.nn.parameter;

export namespace spar::optim {

struct AdamWParameterState final {
  Tensor first_moment;
  Tensor second_moment;
  std::uint64_t step;
};

class AdamW {
public:
  explicit AdamW(std::vector<nn::Parameter> parameters = {}, double learning_rate = 1.0e-3,
                 double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1.0e-8,
                 double weight_decay = 0.01);

  AdamW(const AdamW&) = delete;
  AdamW& operator=(const AdamW&) = delete;
  AdamW(AdamW&&) noexcept = default;
  AdamW& operator=(AdamW&&) noexcept = default;

  void step();
  void zero_grad();
  /// Transactionally stages optimizer moments and moves all tracked Parameters at a graph boundary.
  void move_to(Device target);

  [[nodiscard]] std::size_t parameter_count() const noexcept;
  [[nodiscard]] double learning_rate() const noexcept;
  void set_learning_rate(double learning_rate);
  [[nodiscard]] double beta1() const noexcept;
  [[nodiscard]] double beta2() const noexcept;
  [[nodiscard]] double epsilon() const noexcept;
  [[nodiscard]] double weight_decay() const noexcept;

  [[nodiscard]] bool tracks(const nn::Parameter& parameter) const noexcept;
  [[nodiscard]] std::optional<AdamWParameterState>
  parameter_state(const nn::Parameter& parameter) const;
  void set_parameter_state(const nn::Parameter& parameter,
                           std::optional<AdamWParameterState> state);

private:
  struct State final {
    Tensor first_moment;
    Tensor second_moment;
    std::uint64_t step;
  };

  struct Entry final {
    nn::Parameter parameter;
    std::optional<State> state;
  };

  std::vector<Entry> entries_;
  double learning_rate_;
  double beta1_;
  double beta2_;
  double epsilon_;
  double weight_decay_;
};

} // namespace spar::optim
