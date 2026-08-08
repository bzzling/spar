module spar.ops.elementwise;

import std;
import spar.dtype;
import spar.ops.scalar;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_binary_inputs(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape()) {
    throw invalid_argument{"Elementwise operations require identical shapes"};
  }
  if (a.dtype() != b.dtype()) {
    throw invalid_argument{"Elementwise operations require identical dtypes"};
  }
  if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64) {
    throw invalid_argument{"Elementwise operations currently support floating-point dtypes only"};
  }
  if (!a.is_contiguous() || !b.is_contiguous()) {
    throw invalid_argument{"Elementwise operations currently require contiguous tensors"};
  }
}

template <typename T> Tensor add_values(const Tensor& a, const Tensor& b) {
  Tensor output{a.shape(), a.dtype()};
  const auto a_values{a.span<T>()};
  const auto b_values{b.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = a_values[index] + b_values[index];
  }
  return output;
}

template <typename T> Tensor subtract_values(const Tensor& a, const Tensor& b) {
  Tensor output{a.shape(), a.dtype()};
  const auto a_values{a.span<T>()};
  const auto b_values{b.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = a_values[index] - b_values[index];
  }
  return output;
}

template <typename T> Tensor multiply_values(const Tensor& a, const Tensor& b) {
  Tensor output{a.shape(), a.dtype()};
  const auto a_values{a.span<T>()};
  const auto b_values{b.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = a_values[index] * b_values[index];
  }
  return output;
}

template <typename T> Tensor divide_values(const Tensor& a, const Tensor& b) {
  Tensor output{a.shape(), a.dtype()};
  const auto a_values{a.span<T>()};
  const auto b_values{b.span<T>()};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = a_values[index] / b_values[index];
  }
  return output;
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b) {
  validate_binary_inputs(a, b);
  Tensor output{a.dtype() == DType::Float32 ? add_values<float>(a, b) : add_values<double>(a, b)};
  if (a.requires_grad() || b.requires_grad()) {
    const bool grad_a{a.requires_grad()};
    const bool grad_b{b.requires_grad()};
    vector<Tensor> parents;
    if (grad_a) {
      parents.push_back(a);
    }
    if (grad_b) {
      parents.push_back(b);
    }
    detail::record_operation(output, std::move(parents), [grad_a, grad_b](const Tensor& gradient) {
      vector<Tensor> contributions;
      if (grad_a) {
        contributions.push_back(gradient);
      }
      if (grad_b) {
        contributions.push_back(gradient);
      }
      return contributions;
    });
  }
  return output;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
  validate_binary_inputs(a, b);
  Tensor output{a.dtype() == DType::Float32 ? subtract_values<float>(a, b)
                                            : subtract_values<double>(a, b)};
  if (a.requires_grad() || b.requires_grad()) {
    const bool grad_a{a.requires_grad()};
    const bool grad_b{b.requires_grad()};
    vector<Tensor> parents;
    if (grad_a) {
      parents.push_back(a);
    }
    if (grad_b) {
      parents.push_back(b);
    }
    detail::record_operation(output, std::move(parents), [grad_a, grad_b](const Tensor& gradient) {
      vector<Tensor> contributions;
      if (grad_a) {
        contributions.push_back(gradient);
      }
      if (grad_b) {
        contributions.push_back(multiply_scalar(gradient, -1.0));
      }
      return contributions;
    });
  }
  return output;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
  validate_binary_inputs(a, b);
  Tensor output{a.dtype() == DType::Float32 ? multiply_values<float>(a, b)
                                            : multiply_values<double>(a, b)};
  if (a.requires_grad() || b.requires_grad()) {
    const bool grad_a{a.requires_grad()};
    const bool grad_b{b.requires_grad()};
    optional<Tensor> saved_a;
    optional<Tensor> saved_b;
    if (grad_b) {
      saved_a.emplace(a.detach().clone());
    }
    if (grad_a) {
      saved_b.emplace(b.detach().clone());
    }
    vector<Tensor> parents;
    if (grad_a) {
      parents.push_back(a);
    }
    if (grad_b) {
      parents.push_back(b);
    }
    detail::record_operation(output, std::move(parents),
                             [grad_a, grad_b, saved_a, saved_b](const Tensor& gradient) {
                               vector<Tensor> contributions;
                               if (grad_a) {
                                 contributions.push_back(multiply(gradient, *saved_b));
                               }
                               if (grad_b) {
                                 contributions.push_back(multiply(gradient, *saved_a));
                               }
                               return contributions;
                             });
  }
  return output;
}

Tensor divide(const Tensor& a, const Tensor& b) {
  validate_binary_inputs(a, b);
  Tensor output{a.dtype() == DType::Float32 ? divide_values<float>(a, b)
                                            : divide_values<double>(a, b)};
  if (a.requires_grad() || b.requires_grad()) {
    const bool grad_a{a.requires_grad()};
    const bool grad_b{b.requires_grad()};
    optional<Tensor> saved_a;
    optional<Tensor> saved_b;
    if (grad_b) {
      saved_a.emplace(a.detach().clone());
    }
    saved_b.emplace(b.detach().clone());

    vector<Tensor> parents;
    if (grad_a) {
      parents.push_back(a);
    }
    if (grad_b) {
      parents.push_back(b);
    }
    detail::record_operation(
        output, std::move(parents), [grad_a, grad_b, saved_a, saved_b](const Tensor& gradient) {
          vector<Tensor> contributions;
          if (grad_a) {
            contributions.push_back(divide(gradient, *saved_b));
          }
          if (grad_b) {
            const auto denominator{multiply(*saved_b, *saved_b)};
            const auto numerator{multiply(gradient, *saved_a)};
            contributions.push_back(multiply_scalar(divide(numerator, denominator), -1.0));
          }
          return contributions;
        });
  }
  return output;
}

} // namespace spar
