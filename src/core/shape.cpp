module spar.shape;

import std;

using namespace std;

namespace spar {

Shape::Shape() = default;

Shape::Shape(initializer_list<dimension_type> dimensions)
    : Shape{vector<dimension_type>{dimensions}} {}

Shape::Shape(vector<dimension_type> dimensions)
    : dimensions_{std::move(dimensions)}, numel_{compute_numel(dimensions_)} {}

size_t Shape::rank() const noexcept {
  return dimensions_.size();
}

Shape::dimension_type Shape::operator[](size_t index) const {
  return dimensions_.at(index);
}

span<const Shape::dimension_type> Shape::dimensions() const noexcept {
  return dimensions_;
}

size_t Shape::numel() const noexcept {
  return numel_;
}

vector<Shape::stride_type> Shape::contiguous_strides() const {
  vector<stride_type> strides(rank());
  stride_type running{1};

  for (size_t index{rank()}; index > 0; --index) {
    const size_t current{index - 1};
    strides[current] = running;
    const auto wide_dimension{static_cast<uint64_t>(dimensions_[current])};
    if (wide_dimension > numeric_limits<stride_type>::max()) {
      throw overflow_error{"Shape dimension exceeds the platform stride range"};
    }
    const auto dimension{static_cast<stride_type>(wide_dimension)};
    if (dimension != 0 && running > numeric_limits<stride_type>::max() / dimension) {
      throw overflow_error{"Shape stride calculation overflow"};
    }
    running *= dimension;
  }
  return strides;
}

size_t Shape::compute_numel(const vector<dimension_type>& dimensions) {
  for (const auto dimension : dimensions) {
    if (dimension < 0) {
      throw invalid_argument{"Shape dimensions cannot be negative"};
    }
  }

  if (ranges::find(dimensions, dimension_type{0}) != dimensions.end()) {
    return 0;
  }

  size_t result{1};
  for (const auto dimension : dimensions) {
    const auto wide_value{static_cast<uint64_t>(dimension)};
    if (wide_value > numeric_limits<size_t>::max()) {
      throw overflow_error{"Shape dimension exceeds the platform size range"};
    }
    const auto value{static_cast<size_t>(wide_value)};
    if (value != 0 && result > numeric_limits<size_t>::max() / value) {
      throw overflow_error{"Shape element count overflow"};
    }
    result *= value;
  }
  return result;
}

} // namespace spar
