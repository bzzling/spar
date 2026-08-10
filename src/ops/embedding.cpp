module spar.ops.embedding;

import std;
import spar.dtype;
import spar.shape;
import spar.tensor;

using namespace std;

namespace spar {
namespace {

void validate_embedding_inputs(const Tensor& weight, const Tensor& indices) {
  detail::validate_same_device(weight, indices, "embedding_lookup");
  if (weight.rank() != 2) {
    throw invalid_argument{"embedding_lookup weight must have rank 2"};
  }
  if (weight.dtype() != DType::Float32 && weight.dtype() != DType::Float64) {
    throw invalid_argument{"embedding_lookup weight must have a floating-point dtype"};
  }
  if (indices.dtype() != DType::Int32 && indices.dtype() != DType::Int64) {
    throw invalid_argument{"embedding_lookup indices must have Int32 or Int64 dtype"};
  }
}

template <typename Index>
vector<size_t> validated_indices(const Tensor& indices, size_t vocabulary_size) {
  vector<size_t> result;
  result.reserve(indices.numel());
  for (size_t position{0}; position < indices.numel(); ++position) {
    const Index index{detail::logical_value<Index>(indices, position)};
    if (index < 0 || static_cast<uint64_t>(index) >= static_cast<uint64_t>(vocabulary_size)) {
      throw out_of_range{"embedding_lookup index is outside the weight vocabulary"};
    }
    result.push_back(static_cast<size_t>(index));
  }
  return result;
}

vector<size_t> validated_indices(const Tensor& indices, size_t vocabulary_size) {
  if (indices.dtype() == DType::Int32) {
    return validated_indices<int32_t>(indices, vocabulary_size);
  }
  return validated_indices<int64_t>(indices, vocabulary_size);
}

Shape embedding_output_shape(const Tensor& weight, const Tensor& indices) {
  vector<Shape::dimension_type> dimensions{indices.shape().dimensions().begin(),
                                           indices.shape().dimensions().end()};
  dimensions.push_back(weight.shape()[1]);
  return Shape{std::move(dimensions)};
}

template <typename T>
Tensor embedding_values(const Tensor& weight, const Tensor& indices, span<const size_t> tokens) {
  const size_t embedding_dimension{static_cast<size_t>(weight.shape()[1])};
  Tensor output{embedding_output_shape(weight, indices), weight.dtype(), weight.device()};
  auto output_values{output.span<T>()};
  for (size_t position{0}; position < tokens.size(); ++position) {
    for (size_t feature{0}; feature < embedding_dimension; ++feature) {
      output_values[position * embedding_dimension + feature] =
          detail::logical_value<T>(weight, tokens[position] * embedding_dimension + feature);
    }
  }
  return output;
}

template <typename T>
Tensor embedding_gradient(const Tensor& gradient, const Tensor& saved_indices,
                          const Shape& weight_shape) {
  const size_t vocabulary_size{static_cast<size_t>(weight_shape[0])};
  const size_t embedding_dimension{static_cast<size_t>(weight_shape[1])};
  const vector<size_t> tokens{validated_indices(saved_indices, vocabulary_size)};
  detail::validate_same_device(gradient, saved_indices, "embedding backward");
  Tensor contribution{zeros(weight_shape, gradient.dtype(), gradient.device())};
  auto contribution_values{contribution.span<T>()};
  for (size_t position{0}; position < tokens.size(); ++position) {
    for (size_t feature{0}; feature < embedding_dimension; ++feature) {
      contribution_values[tokens[position] * embedding_dimension + feature] +=
          detail::logical_value<T>(gradient, position * embedding_dimension + feature);
    }
  }
  return contribution;
}

} // namespace

Tensor embedding_lookup(const Tensor& weight, const Tensor& indices) {
  validate_embedding_inputs(weight, indices);
  const size_t vocabulary_size{static_cast<size_t>(weight.shape()[0])};
  const vector<size_t> tokens{validated_indices(indices, vocabulary_size)};
  Tensor output{weight.dtype() == DType::Float32
                    ? embedding_values<float>(weight, indices, tokens)
                    : embedding_values<double>(weight, indices, tokens)};
  if (weight.requires_grad()) {
    const Tensor saved_indices{indices.detach().clone()};
    const Shape weight_shape{weight.shape()};
    const DType dtype{weight.dtype()};
    detail::record_operation(
        output, {weight}, [saved_indices, weight_shape, dtype](const Tensor& gradient) {
          if (dtype == DType::Float32) {
            return vector<Tensor>{embedding_gradient<float>(gradient, saved_indices, weight_shape)};
          }
          return vector<Tensor>{embedding_gradient<double>(gradient, saved_indices, weight_shape)};
        });
  }
  return output;
}

} // namespace spar
