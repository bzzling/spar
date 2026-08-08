module spar.ops.matmul;

import std;
import spar.dtype;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_matmul_inputs(const Tensor& a, const Tensor& b) {
  if (a.dtype() != b.dtype()) {
    throw invalid_argument{"matmul requires identical dtypes"};
  }
  if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64) {
    throw invalid_argument{"matmul currently supports floating-point dtypes only"};
  }
  if (a.rank() != 2 || b.rank() != 2) {
    throw invalid_argument{"matmul currently requires rank-2 tensors"};
  }
  if (a.shape()[1] != b.shape()[0]) {
    throw invalid_argument{"matmul inner dimensions must match"};
  }
  if (!a.is_contiguous() || !b.is_contiguous()) {
    throw invalid_argument{"matmul currently requires contiguous tensors"};
  }
}

template <typename T> Tensor matmul_values(const Tensor& a, const Tensor& b) {
  const auto m{static_cast<size_t>(a.shape()[0])};
  const auto k_extent{static_cast<size_t>(a.shape()[1])};
  const auto n{static_cast<size_t>(b.shape()[1])};
  Tensor output{Shape{a.shape()[0], b.shape()[1]}, a.dtype()};

  const auto a_values{a.span<T>()};
  const auto b_values{b.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t i{0}; i < m; ++i) {
    for (size_t j{0}; j < n; ++j) {
      T accumulator{0};
      for (size_t k{0}; k < k_extent; ++k) {
        accumulator += a_values[i * k_extent + k] * b_values[k * n + j];
      }
      output_values[i * n + j] = accumulator;
    }
  }
  return output;
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
  validate_matmul_inputs(a, b);
  switch (a.dtype()) {
  case DType::Float32:
    return matmul_values<float>(a, b);
  case DType::Float64:
    return matmul_values<double>(a, b);
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"matmul dtype validation invariant violated"};
  }
  throw logic_error{"matmul dtype validation invariant violated"};
}

} // namespace spar
