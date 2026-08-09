module spar.random;

import std;
import spar.dtype;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar {

Random::Random(uint64_t seed) noexcept : state_{seed} {}

uint64_t Random::state() const noexcept {
  return state_;
}

void Random::set_state(uint64_t state) noexcept {
  state_ = state;
}

uint64_t Random::next_u64() noexcept {
  state_ += 0x9E3779B97F4A7C15ULL;
  auto value{state_};
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

float Random::uniform_float() noexcept {
  constexpr float denominator{16'777'216.0F};
  return static_cast<float>(next_u64() >> 40U) / denominator;
}

double Random::uniform_double() noexcept {
  constexpr double denominator{9'007'199'254'740'992.0};
  return static_cast<double>(next_u64() >> 11U) / denominator;
}

Tensor random_uniform(Shape shape, DType dtype, Random& random, double low, double high) {
  if (!(low < high)) {
    throw invalid_argument{"random_uniform requires low < high"};
  }

  switch (dtype) {
  case DType::Float32:
  case DType::Float64:
    break;
  case DType::Int32:
  case DType::Int64:
    throw invalid_argument{"random_uniform currently supports floating-point dtypes only"};
  default:
    throw invalid_argument{"random_uniform received an unknown dtype"};
  }

  Tensor tensor{std::move(shape), dtype};
  switch (dtype) {
  case DType::Float32: {
    const auto low_value{static_cast<float>(low)};
    const auto width{static_cast<float>(high - low)};
    for (auto& value : tensor.span<float>()) {
      value = low_value + width * random.uniform_float();
    }
    break;
  }
  case DType::Float64: {
    const auto width{high - low};
    for (auto& value : tensor.span<double>()) {
      value = low + width * random.uniform_double();
    }
    break;
  }
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"random_uniform dtype validation invariant violated"};
  default:
    throw logic_error{"random_uniform dtype validation invariant violated"};
  }
  return tensor;
}

} // namespace spar
