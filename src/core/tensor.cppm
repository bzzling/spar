export module spar.tensor;

import std;
export import spar.device;
import spar.dtype;
import spar.shape;
import spar.storage;

export namespace spar {
class Tensor;
}

export namespace spar::detail {
struct AutogradAccess;
struct AutogradMeta;
struct CudaTensorAccess;
template <typename T> T logical_value(const spar::Tensor& tensor, std::size_t logical_index);
void copy_tensor_values(spar::Tensor& destination, const spar::Tensor& source);
void swap_storage_payloads(spar::Tensor& original, spar::Tensor& staged) noexcept;
} // namespace spar::detail

export namespace spar {

/// A typed, shaped, strided view over reference-counted Storage.
/// Tensor copies and metadata-only views share Storage; use `clone()` for an independent copy.
class Tensor {
public:
  /// allocates contiguous row-major storage without initializing its elements.
  Tensor(Shape shape, DType dtype, Device device = Device::cpu());
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
  [[nodiscard]] Device device() const noexcept;
  /// returns logical tensor bytes, not the size of the shared allocation.
  [[nodiscard]] std::size_t nbytes() const noexcept;
  [[nodiscard]] bool is_contiguous() const noexcept;

  [[nodiscard]] bool requires_grad() const noexcept;
  /// A Tensor identity is a leaf when it has no differentiable producing operation.
  [[nodiscard]] bool is_leaf() const noexcept;
  [[nodiscard]] bool has_grad() const noexcept;
  /// Changes gradient tracking on a leaf identity; only floating tensors may enable it.
  void set_requires_grad(bool enabled = true);
  /// Returns the accumulated leaf gradient or throws when none exists.
  [[nodiscard]] Tensor grad() const;
  /// Clears the accumulated gradient shared by all handles of this leaf identity.
  void zero_grad();
  /// Runs first-order reverse-mode differentiation from a one-element loss.
  void backward();
  /// Shares value Storage while creating a fresh leaf identity with no graph history.
  /// Mutations through a detached alias are explicit and are not version-checked yet.
  [[nodiscard]] Tensor detach() const;
  /// Transfers logical values synchronously; a same-Device transfer preserves Tensor identity.
  [[nodiscard]] Tensor to(Device target) const;

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
  /// returns a right-aligned metadata-only broadcast view using zero strides.
  [[nodiscard]] Tensor expand(Shape target_shape) const;

  /// returns mutable typed access for a contiguous non-grad identity with a matching dtype.
  template <typename T> [[nodiscard]] std::span<T> span() & {
    if (requires_grad()) {
      throw std::logic_error{
          "Mutable tensor access requires a non-grad Tensor identity; detach explicitly first"};
    }
    validate_access<T>();
    if (numel() == 0) {
      return {};
    }
    return {reinterpret_cast<T*>(mutable_host_data()) + storage_offset_, numel()};
  }

  /// returns read-only typed access after validating dtype and logical contiguity.
  template <typename T> [[nodiscard]] std::span<const T> span() const& {
    validate_access<T>();
    if (numel() == 0) {
      return {};
    }
    return {reinterpret_cast<const T*>(host_data()) + storage_offset_, numel()};
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
  friend struct detail::AutogradAccess;
  friend struct detail::CudaTensorAccess;
  template <typename T>
  friend T detail::logical_value(const Tensor& tensor, std::size_t logical_index);
  friend void detail::copy_tensor_values(Tensor& destination, const Tensor& source);
  friend void detail::swap_storage_payloads(Tensor& original, Tensor& staged) noexcept;
  friend Tensor zeros(Shape shape, DType dtype, Device device);

  Tensor(std::shared_ptr<detail::Storage> storage, DType dtype, Shape shape,
         std::vector<Shape::stride_type> strides, std::size_t storage_offset);

  void initialize_autograd();
  void reset_to_empty() noexcept;
  void validate_view_bounds() const;

  [[nodiscard]] static std::size_t checked_nbytes(std::size_t numel, DType dtype);
  [[nodiscard]] std::size_t checked_storage_byte_offset() const;
  [[nodiscard]] Tensor materialize_contiguous() const;
  [[nodiscard]] Tensor materialize_values(Device target) const;
  [[nodiscard]] Tensor materialize_cpu_logical() const;
  [[nodiscard]] std::size_t logical_storage_index(std::size_t logical_index) const;
  [[nodiscard]] std::byte* mutable_host_data();
  [[nodiscard]] const std::byte* host_data() const;

  template <typename T> void validate_access() const {
    if (!device().is_cpu()) {
      throw std::logic_error{"Tensor span requires CPU Storage"};
    }
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
  std::shared_ptr<detail::AutogradMeta> autograd_;
};

void swap(Tensor& left, Tensor& right) noexcept;

/// returns a contiguous tensor initialized to zero.
[[nodiscard]] Tensor zeros(Shape shape, DType dtype, Device device = Device::cpu());
/// returns a contiguous tensor initialized to one.
[[nodiscard]] Tensor ones(Shape shape, DType dtype, Device device = Device::cpu());

/// returns a contiguous tensor filled with `value`, inferring its dtype from `T`.
template <typename T>
[[nodiscard]] Tensor full(Shape shape, T value, Device device = Device::cpu()) {
  if (device.is_cpu()) {
    Tensor tensor{std::move(shape), dtype_of<T>(), device};
    tensor.fill<T>(value);
    return tensor;
  }
  Tensor staging{std::move(shape), dtype_of<T>(), Device::cpu()};
  staging.fill<T>(value);
  return staging.to(device);
}

} // namespace spar

export namespace spar::detail {

struct CudaTensorAccess final {
  [[nodiscard]] static void* mutable_data(Tensor& tensor);
  [[nodiscard]] static const void* data(const Tensor& tensor);
};

template <typename T> T logical_value(const Tensor& tensor, std::size_t logical_index) {
  if (dtype_of<T>() != tensor.dtype_) {
    throw std::logic_error{"Internal logical tensor access dtype mismatch"};
  }
  if (logical_index >= tensor.numel()) {
    throw std::out_of_range{"Internal logical tensor index is out of range"};
  }
  const auto base{reinterpret_cast<const T*>(tensor.host_data())};
  return base[tensor.logical_storage_index(logical_index)];
}

[[nodiscard]] Shape broadcast_shape(const Shape& left, const Shape& right);
void validate_same_device(const Tensor& left, const Tensor& right, std::string_view operation);
void require_cpu(const Tensor& tensor, std::string_view operation);
void copy_tensor_values(Tensor& destination, const Tensor& source);
/// Commits a pre-staged placement change without replacing Tensor/Storage identities.
void swap_storage_payloads(Tensor& original, Tensor& staged) noexcept;
[[nodiscard]] Tensor reduce_gradient_to_shape(const Tensor& gradient, const Shape& original_shape);

/// Internal operation-recording hook; graph node types remain private to spar.tensor.
using BackwardFunction = std::function<std::vector<Tensor>(const Tensor&)>;

void record_operation(Tensor& output, std::vector<Tensor> requiring_parents,
                      BackwardFunction backward);
void record_transfer_operation(Tensor& output, Tensor requiring_parent, BackwardFunction backward);

/// Internal semantic-identity predicate used by value-semantic Parameter handles.
[[nodiscard]] bool shares_autograd_identity(const Tensor& left, const Tensor& right) noexcept;

} // namespace spar::detail
