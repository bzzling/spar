module spar.shape;

import std;

using namespace std;

namespace spar {

Shape::Shape() = default;

Shape::Shape(initializer_list<dimension_type> dimensions)
    : Shape{vector<dimension_type>{dimensions}} {}

Shape::Shape(vector<dimension_type> dimensions)
    : dimensions_{std::move(dimensions)}, numel_{compute_numel(dimensions_)} {}

Shape::Shape(Shape&& other) noexcept
    : dimensions_{std::move(other.dimensions_)}, numel_{other.numel_} {
  other.dimensions_.clear();
  other.numel_ = 0;
}

Shape& Shape::operator=(Shape&& other) noexcept {
  if (this != &other) {
    dimensions_ = std::move(other.dimensions_);
    numel_ = other.numel_;
    other.dimensions_.clear();
    other.numel_ = 0;
  }
  return *this;
}

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

Shape::stride_type Shape::checked_extent(dimension_type dimension) {
  if (dimension < 0) {
    throw invalid_argument{"Shape dimensions cannot be negative"};
  }

  const auto wide_dimension{static_cast<uint64_t>(dimension)};
  if (wide_dimension > numeric_limits<stride_type>::max()) {
    throw overflow_error{"Shape dimension exceeds the platform size range"};
  }
  return static_cast<stride_type>(wide_dimension);
}

vector<Shape::stride_type> Shape::contiguous_strides() const {
  vector<stride_type> strides(rank());
  stride_type running{1};

  for (size_t index{rank()}; index > 0; --index) {
    const size_t current{index - 1};
    strides[current] = running;
    const auto dimension{checked_extent(dimensions_[current])};
    if (dimension != 0 && running > numeric_limits<stride_type>::max() / dimension) {
      throw overflow_error{"Shape stride calculation overflow"};
    }
    running *= dimension;
  }
  return strides;
}

size_t Shape::compute_numel(const vector<dimension_type>& dimensions) {
  bool has_zero_extent{false};
  for (const auto dimension : dimensions) {
    if (checked_extent(dimension) == 0) {
      has_zero_extent = true;
    }
  }

  if (has_zero_extent) {
    return 0;
  }

  size_t result{1};
  for (const auto dimension : dimensions) {
    const auto value{static_cast<size_t>(dimension)};
    if (result > numeric_limits<size_t>::max() / value) {
      throw overflow_error{"Shape element count overflow"};
    }
    result *= value;
  }
  return result;
}

} // namespace spar
