export module spar.tensor;

import std;
import spar.dtype;
import spar.shape;
import spar.storage;

export namespace spar {

/// A typed, shaped, strided view over reference-counted CPU Storage.
/// Tensor copies and metadata-only views share Storage; use `clone()` for an independent copy.
class Tensor {
public:
  /// allocates contiguous row-major storage without initializing its elements.
  Tensor(Shape shape, DType dtype);
  /// makes a shallow handle copy sharing the same Storage.
  Tensor(const Tensor&) = default;
  Tensor& operator=(const Tensor&) = default;
  /// transfers Storage and metadata, leaving the source coherently empty.
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;
  ~Tensor();

  void swap(Tensor& other) noexcept;

  [[nodiscard]] std::size_t rank() const noexcept;
  [[nodiscard]] const Shape& shape() const noexcept;
  [[nodiscard]] std::span<const Shape::stride_type> strides() const noexcept;
  [[nodiscard]] std::size_t numel() const noexcept;
  [[nodiscard]] DType dtype() const noexcept;
  /// returns logical tensor bytes, not the size of the shared allocation.
  [[nodiscard]] std::size_t nbytes() const noexcept;
  [[nodiscard]] bool is_contiguous() const noexcept;

  /// always returns an independent contiguous row-major copy in logical element order.
  [[nodiscard]] Tensor clone() const;
  /// shares Storage when already contiguous; otherwise materializes logical values.
  [[nodiscard]] Tensor contiguous() const;
  /// returns a metadata-only contiguous view; non-contiguous inputs are rejected.
  [[nodiscard]] Tensor reshape(Shape new_shape) const;
  /// returns a metadata-only view with two dimensions and strides exchanged.
  [[nodiscard]] Tensor transpose(std::size_t axis_a, std::size_t axis_b) const;
  /// returns a metadata-only view with dimensions and strides reordered by `axes`.
  [[nodiscard]] Tensor permute(std::span<const std::size_t> axes) const;
  [[nodiscard]] Tensor permute(std::initializer_list<std::size_t> axes) const;

  /// returns mutable typed access after validating dtype and logical contiguity.
  template <typename T> [[nodiscard]] std::span<T> span() & {
    validate_access<T>();
    if (numel() == 0) {
      return {};
    }
    return {reinterpret_cast<T*>(mutable_data()) + storage_offset_, numel()};
  }

  /// returns read-only typed access after validating dtype and logical contiguity.
  template <typename T> [[nodiscard]] std::span<const T> span() const& {
    validate_access<T>();
    if (numel() == 0) {
      return {};
    }
    return {reinterpret_cast<const T*>(data()) + storage_offset_, numel()};
  }

  /// prevents spans that would immediately dangle from temporary Tensor handles.
  template <typename T> std::span<T> span() && = delete;
  template <typename T> std::span<const T> span() const&& = delete;

  /// fills every element of a contiguous tensor; `T` must match the tensor dtype.
  template <typename T> void fill(T value) & {
    auto values{span<T>()};
    std::ranges::fill(values, value);
  }

private:
  Tensor(std::shared_ptr<detail::Storage> storage, DType dtype, Shape shape,
         std::vector<Shape::stride_type> strides, std::size_t storage_offset);

  void reset_to_empty() noexcept;
  void validate_view_bounds() const;

  [[nodiscard]] static std::size_t checked_nbytes(std::size_t numel, DType dtype);
  [[nodiscard]] std::size_t logical_storage_index(std::size_t logical_index) const;
  [[nodiscard]] std::byte* mutable_data() noexcept;
  [[nodiscard]] const std::byte* data() const noexcept;

  template <typename T> void validate_access() const {
    if (dtype_of<T>() != dtype_) {
      throw std::invalid_argument{"Typed tensor access does not match the tensor dtype"};
    }
    if (!is_contiguous()) {
      throw std::invalid_argument{"Tensor span requires a contiguous tensor"};
    }
  }

  std::shared_ptr<detail::Storage> storage_;
  DType dtype_;
  Shape shape_;
  std::vector<Shape::stride_type> strides_;
  std::size_t storage_offset_;
  std::size_t nbytes_;
};

void swap(Tensor& left, Tensor& right) noexcept;

/// returns a contiguous tensor initialized to zero.
[[nodiscard]] Tensor zeros(Shape shape, DType dtype);
/// returns a contiguous tensor initialized to one.
[[nodiscard]] Tensor ones(Shape shape, DType dtype);

/// returns a contiguous tensor filled with `value`, inferring its dtype from `T`.
template <typename T> [[nodiscard]] Tensor full(Shape shape, T value) {
  Tensor tensor{std::move(shape), dtype_of<T>()};
  tensor.fill<T>(value);
  return tensor;
}

} // namespace spar
