export module spar.random;

import std;
import spar.dtype;
import spar.shape;
import spar.tensor;

export namespace spar {

/// explicit deterministic random-number-generator state.
class Random {
public:
  /// creates a generator with the given seed.
  explicit Random(std::uint64_t seed) noexcept;

  [[nodiscard]] std::uint64_t state() const noexcept;
  void set_state(std::uint64_t state) noexcept;

  /// advances and returns the next SplitMix64 output.
  [[nodiscard]] std::uint64_t next_u64() noexcept;
  /// returns a reproducible float in [0, 1).
  [[nodiscard]] float uniform_float() noexcept;
  /// returns a reproducible double in [0, 1).
  [[nodiscard]] double uniform_double() noexcept;

private:
  std::uint64_t state_;
};

/// returns a floating-point tensor sampled uniformly from [low, high).
[[nodiscard]] Tensor random_uniform(Shape shape, DType dtype, Random& random, double low = 0.0,
                                    double high = 1.0);

} // namespace spar
