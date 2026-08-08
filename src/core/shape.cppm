export module spar.shape;

import std;

export namespace spar {

/// a validated dense-tensor shape.
/// dimensions are signed so negative inputs can be diagnosed.
/// `Shape{}` is a rank-zero scalar shape containing one element; zero extents are permitted.
class Shape {
public:
  using dimension_type = std::int64_t;
  using stride_type = std::size_t;

  /// constructs the rank-zero scalar shape.
  Shape();

  /// constructs a shape from brace-enclosed dimensions.
  Shape(std::initializer_list<dimension_type> dimensions);

  /// constructs a shape from an owned dimension vector.
  explicit Shape(std::vector<dimension_type> dimensions);

  Shape(const Shape&) = default;
  Shape& operator=(const Shape&) = default;
  Shape(Shape&& other) noexcept;
  Shape& operator=(Shape&& other) noexcept;

  /// returns the number of dimensions.
  [[nodiscard]] std::size_t rank() const noexcept;

  /// returns one dimension, throwing `std::out_of_range` for an invalid index.
  [[nodiscard]] dimension_type operator[](std::size_t index) const;

  /// returns all dimensions in outermost-to-innermost order.
  [[nodiscard]] std::span<const dimension_type> dimensions() const noexcept;

  /// returns the checked product of the dimensions.
  [[nodiscard]] std::size_t numel() const noexcept;

  /// computes element strides for contiguous row-major storage.
  [[nodiscard]] std::vector<stride_type> contiguous_strides() const;

  friend bool operator==(const Shape&, const Shape&) = default;

private:
  [[nodiscard]] static stride_type checked_extent(dimension_type dimension);
  [[nodiscard]] static std::size_t compute_numel(const std::vector<dimension_type>& dimensions);

  std::vector<dimension_type> dimensions_;
  std::size_t numel_{1};
};

} // namespace spar
