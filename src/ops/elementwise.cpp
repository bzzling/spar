module spar.ops.elementwise;

import std;
import spar.dtype;
import spar.ops.scalar;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

Shape validate_binary_inputs(const Tensor& a, const Tensor& b) {
  detail::validate_same_device(a, b, "Elementwise operation");
  if (a.dtype() != b.dtype()) {
    throw invalid_argument{"Elementwise operations require identical dtypes"};
  }
  if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64) {
    throw invalid_argument{"Elementwise operations currently support floating-point dtypes only"};
  }
  return detail::broadcast_shape(a.shape(), b.shape());
}

template <typename T, typename Operation>
Tensor binary_values(const Tensor& a, const Tensor& b, const Shape& output_shape,
                     Operation operation) {
  Tensor output{output_shape, a.dtype(), a.device()};
  const auto expanded_a{a.detach().expand(output_shape)};
  const auto expanded_b{b.detach().expand(output_shape)};
  auto output_values{output.span<T>()};

  for (size_t index{0}; index < output_values.size(); ++index) {
    output_values[index] = operation(detail::logical_value<T>(expanded_a, index),
                                     detail::logical_value<T>(expanded_b, index));
  }
  return output;
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b) {
  const Shape output_shape{validate_binary_inputs(a, b)};
  Tensor output{a.dtype() == DType::Float32
                    ? binary_values<float>(a, b, output_shape,
                                           [](float left, float right) { return left + right; })
                    : binary_values<double>(a, b, output_shape, [](double left, double right) {
                        return left + right;
                      })};
  if (a.requires_grad() || b.requires_grad()) {
    const bool grad_a{a.requires_grad()};
    const bool grad_b{b.requires_grad()};
    const Shape shape_a{a.shape()};
    const Shape shape_b{b.shape()};
    vector<Tensor> parents;
    if (grad_a) {
      parents.push_back(a);
    }
    if (grad_b) {
      parents.push_back(b);
    }
    detail::record_operation(
        output, std::move(parents), [grad_a, grad_b, shape_a, shape_b](const Tensor& gradient) {
          vector<Tensor> contributions;
          if (grad_a) {
            contributions.push_back(detail::reduce_gradient_to_shape(gradient, shape_a));
          }
          if (grad_b) {
            contributions.push_back(detail::reduce_gradient_to_shape(gradient, shape_b));
          }
          return contributions;
        });
  }
  return output;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
  const Shape output_shape{validate_binary_inputs(a, b)};
  Tensor output{a.dtype() == DType::Float32
                    ? binary_values<float>(a, b, output_shape,
                                           [](float left, float right) { return left - right; })
                    : binary_values<double>(a, b, output_shape, [](double left, double right) {
                        return left - right;
                      })};
  if (a.requires_grad() || b.requires_grad()) {
    const bool grad_a{a.requires_grad()};
    const bool grad_b{b.requires_grad()};
    const Shape shape_a{a.shape()};
    const Shape shape_b{b.shape()};
    vector<Tensor> parents;
    if (grad_a) {
      parents.push_back(a);
    }
    if (grad_b) {
      parents.push_back(b);
    }
    detail::record_operation(
        output, std::move(parents), [grad_a, grad_b, shape_a, shape_b](const Tensor& gradient) {
          vector<Tensor> contributions;
          if (grad_a) {
            contributions.push_back(detail::reduce_gradient_to_shape(gradient, shape_a));
          }
          if (grad_b) {
            const auto negative{multiply_scalar(gradient, -1.0)};
            contributions.push_back(detail::reduce_gradient_to_shape(negative, shape_b));
          }
          return contributions;
        });
  }
  return output;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
  const Shape output_shape{validate_binary_inputs(a, b)};
  Tensor output{a.dtype() == DType::Float32
                    ? binary_values<float>(a, b, output_shape,
                                           [](float left, float right) { return left * right; })
                    : binary_values<double>(a, b, output_shape, [](double left, double right) {
                        return left * right;
                      })};
  if (a.requires_grad() || b.requires_grad()) {
    const bool grad_a{a.requires_grad()};
    const bool grad_b{b.requires_grad()};
    const Shape shape_a{a.shape()};
    const Shape shape_b{b.shape()};
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
    detail::record_operation(
        output, std::move(parents),
        [grad_a, grad_b, shape_a, shape_b, saved_a, saved_b](const Tensor& gradient) {
          vector<Tensor> contributions;
          if (grad_a) {
            const auto local{multiply(gradient, *saved_b)};
            contributions.push_back(detail::reduce_gradient_to_shape(local, shape_a));
          }
          if (grad_b) {
            const auto local{multiply(gradient, *saved_a)};
            contributions.push_back(detail::reduce_gradient_to_shape(local, shape_b));
          }
          return contributions;
        });
  }
  return output;
}

Tensor divide(const Tensor& a, const Tensor& b) {
  const Shape output_shape{validate_binary_inputs(a, b)};
  Tensor output{a.dtype() == DType::Float32
                    ? binary_values<float>(a, b, output_shape,
                                           [](float left, float right) { return left / right; })
                    : binary_values<double>(a, b, output_shape, [](double left, double right) {
                        return left / right;
                      })};
  if (a.requires_grad() || b.requires_grad()) {
    const bool grad_a{a.requires_grad()};
    const bool grad_b{b.requires_grad()};
    const Shape shape_a{a.shape()};
    const Shape shape_b{b.shape()};
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
        output, std::move(parents),
        [grad_a, grad_b, shape_a, shape_b, saved_a, saved_b](const Tensor& gradient) {
          vector<Tensor> contributions;
          if (grad_a) {
            const auto local{divide(gradient, *saved_b)};
            contributions.push_back(detail::reduce_gradient_to_shape(local, shape_a));
          }
          if (grad_b) {
            const auto denominator{multiply(*saved_b, *saved_b)};
            const auto numerator{multiply(gradient, *saved_a)};
            const auto local{multiply_scalar(divide(numerator, denominator), -1.0)};
            contributions.push_back(detail::reduce_gradient_to_shape(local, shape_b));
          }
          return contributions;
        });
  }
  return output;
}

} // namespace spar
