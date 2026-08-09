export module spar.optim.adamw;

import std;
export import spar.nn.parameter;

export namespace spar::optim {

class AdamW {
public:
  explicit AdamW(std::vector<nn::Parameter> parameters = {}, double learning_rate = 1.0e-3,
                 double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1.0e-8,
                 double weight_decay = 0.01);

  void step();
  void zero_grad();

  [[nodiscard]] std::size_t parameter_count() const noexcept;
  [[nodiscard]] double learning_rate() const noexcept;
  void set_learning_rate(double learning_rate);
  [[nodiscard]] double beta1() const noexcept;
  [[nodiscard]] double beta2() const noexcept;
  [[nodiscard]] double epsilon() const noexcept;
  [[nodiscard]] double weight_decay() const noexcept;

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
