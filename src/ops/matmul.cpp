module spar.ops.matmul;

import std;
import spar.dtype;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

struct MatmulPlan final {
  Shape output_shape;
  Shape expanded_a_shape;
  Shape expanded_b_shape;
  size_t m;
  size_t k;
  size_t n;
};

MatmulPlan plan_matmul(const Tensor& a, const Tensor& b) {
  detail::validate_same_device(a, b, "matmul");
  if (a.rank() < 2 || b.rank() < 2) {
    throw invalid_argument{"matmul requires tensors with rank at least 2"};
  }
  if (a.dtype() != b.dtype()) {
    throw invalid_argument{"matmul requires identical dtypes"};
  }
  if (a.dtype() != DType::Float32 && a.dtype() != DType::Float64) {
    throw invalid_argument{"matmul currently supports floating-point dtypes only"};
  }
  if (a.shape()[a.rank() - 1] != b.shape()[b.rank() - 2]) {
    throw invalid_argument{"matmul inner dimensions must match"};
  }

  vector<Shape::dimension_type> a_batch_dimensions;
  a_batch_dimensions.reserve(a.rank() - 2);
  for (size_t axis{0}; axis + 2 < a.rank(); ++axis) {
    a_batch_dimensions.push_back(a.shape()[axis]);
  }
  vector<Shape::dimension_type> b_batch_dimensions;
  b_batch_dimensions.reserve(b.rank() - 2);
  for (size_t axis{0}; axis + 2 < b.rank(); ++axis) {
    b_batch_dimensions.push_back(b.shape()[axis]);
  }

  const Shape batch_shape{detail::broadcast_shape(Shape{std::move(a_batch_dimensions)},
                                                  Shape{std::move(b_batch_dimensions)})};
  vector<Shape::dimension_type> output_dimensions{batch_shape.dimensions().begin(),
                                                  batch_shape.dimensions().end()};
  vector<Shape::dimension_type> expanded_a_dimensions{output_dimensions};
  vector<Shape::dimension_type> expanded_b_dimensions{output_dimensions};
  const auto m_extent{a.shape()[a.rank() - 2]};
  const auto k_extent{a.shape()[a.rank() - 1]};
  const auto n_extent{b.shape()[b.rank() - 1]};
  output_dimensions.push_back(m_extent);
  output_dimensions.push_back(n_extent);
  expanded_a_dimensions.push_back(m_extent);
  expanded_a_dimensions.push_back(k_extent);
  expanded_b_dimensions.push_back(k_extent);
  expanded_b_dimensions.push_back(n_extent);

  return MatmulPlan{
      Shape{std::move(output_dimensions)},     Shape{std::move(expanded_a_dimensions)},
      Shape{std::move(expanded_b_dimensions)}, static_cast<size_t>(m_extent),
      static_cast<size_t>(k_extent),           static_cast<size_t>(n_extent)};
}

template <typename T>
Tensor matmul_values(const Tensor& a, const Tensor& b, const MatmulPlan& plan) {
  Tensor output{plan.output_shape, a.dtype(), a.device()};
  if (output.numel() == 0) {
    return output;
  }

  const Tensor expanded_a{a.detach().expand(plan.expanded_a_shape)};
  const Tensor expanded_b{b.detach().expand(plan.expanded_b_shape)};
  auto output_values{output.span<T>()};
  const size_t matrix_output_size{plan.m * plan.n};
  const size_t batch_count{output.numel() / matrix_output_size};

  for (size_t batch{0}; batch < batch_count; ++batch) {
    for (size_t m_index{0}; m_index < plan.m; ++m_index) {
      for (size_t n_index{0}; n_index < plan.n; ++n_index) {
        T accumulator{0};
        for (size_t k_index{0}; k_index < plan.k; ++k_index) {
          const size_t a_index{(batch * plan.m + m_index) * plan.k + k_index};
          const size_t b_index{(batch * plan.k + k_index) * plan.n + n_index};
          accumulator += detail::logical_value<T>(expanded_a, a_index) *
                         detail::logical_value<T>(expanded_b, b_index);
        }
        output_values[(batch * plan.m + m_index) * plan.n + n_index] = accumulator;
      }
    }
  }
  return output;
}

Tensor transpose_last_two(const Tensor& tensor) {
  vector<size_t> axes(tensor.rank());
  for (size_t axis{0}; axis < tensor.rank(); ++axis) {
    axes[axis] = axis;
  }
  std::swap(axes[tensor.rank() - 2], axes[tensor.rank() - 1]);
  return tensor.permute(axes);
}

} // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
  const MatmulPlan plan{plan_matmul(a, b)};
  Tensor output{a.dtype() == DType::Float32 ? matmul_values<float>(a, b, plan)
                                            : matmul_values<double>(a, b, plan)};
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
            const Tensor expanded_gradient{matmul(gradient, transpose_last_two(*saved_b))};
            contributions.push_back(detail::reduce_gradient_to_shape(expanded_gradient, shape_a));
          }
          if (grad_b) {
            const Tensor expanded_gradient{matmul(transpose_last_two(*saved_a), gradient)};
            contributions.push_back(detail::reduce_gradient_to_shape(expanded_gradient, shape_b));
          }
          return contributions;
        });
  }
  return output;
}

} // namespace spar
