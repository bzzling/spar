export module spar.tensor;

import std;
import spar.dtype;
import spar.shape;

export namespace spar {

/// an owning, contiguous, row-major CPU tensor.
/// construction allocates storage without initializing its elements.
/// use `zeros`, `ones`, or `full` when initialized values are required.
class Tensor {
public:
  /// allocates storage for `shape` and `dtype` without initializing it.
  Tensor(Shape shape, DType dtype);
  /// makes an independent deep copy.
  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);
  /// transfers storage in constant time and leaves `other` empty.
  Tensor(Tensor&&) noexcept;
  Tensor& operator=(Tensor&&) noexcept;
  ~Tensor();

  void swap(Tensor& other) noexcept;

  [[nodiscard]] std::size_t rank() const noexcept;
  [[nodiscard]] const Shape& shape() const noexcept;
  [[nodiscard]] std::span<const Shape::stride_type> strides() const noexcept;
  [[nodiscard]] std::size_t numel() const noexcept;
  [[nodiscard]] DType dtype() const noexcept;
  [[nodiscard]] std::size_t nbytes() const noexcept;

  /// returns mutable typed access after validating `T` against `dtype()`.
  template <typename T> [[nodiscard]] std::span<T> span() {
    validate_type<T>();
    return {reinterpret_cast<T*>(storage_.get()), numel()};
  }

  /// returns read-only typed access after validating `T` against `dtype()`.
  template <typename T> [[nodiscard]] std::span<const T> span() const {
    validate_type<T>();
    return {reinterpret_cast<const T*>(storage_.get()), numel()};
  }

  /// fills every element with `value`; `T` must match the tensor dtype.
  template <typename T> void fill(T value) {
    auto values{span<T>()};
    std::ranges::fill(values, value);
  }

private:
  void reset_to_empty() noexcept;

  [[nodiscard]] static std::size_t checked_nbytes(std::size_t numel, DType dtype);
  [[nodiscard]] static std::unique_ptr<std::byte[]> allocate(std::size_t nbytes);

  template <typename T> void validate_type() const {
    if (dtype_of<T>() != dtype_) {
      throw std::invalid_argument{"Typed tensor access does not match the tensor dtype"};
    }
  }

  DType dtype_;
  Shape shape_;
  std::vector<Shape::stride_type> strides_;
  std::size_t nbytes_;
  std::unique_ptr<std::byte[]> storage_;
};

void swap(Tensor& left, Tensor& right) noexcept;

/// returns a tensor initialized to zero.
[[nodiscard]] Tensor zeros(Shape shape, DType dtype);
/// returns a tensor initialized to one.
[[nodiscard]] Tensor ones(Shape shape, DType dtype);

/// returns a tensor filled with `value`, inferring its dtype from `T`.
template <typename T> [[nodiscard]] Tensor full(Shape shape, T value) {
  Tensor tensor{std::move(shape), dtype_of<T>()};
  tensor.fill<T>(value);
  return tensor;
}

} // namespace spar
