module spar.tensor;

import std;
import spar.device;
import spar.dtype;
import spar.shape;
import spar.storage;

using namespace std;

namespace spar {

Tensor::Tensor(Shape shape, DType dtype, Device device)
    : storage_{}, dtype_{dtype}, shape_{std::move(shape)}, strides_{shape_.contiguous_strides()},
      storage_offset_{0}, nbytes_{checked_nbytes(shape_.numel(), dtype_)} {
  storage_ = make_shared<detail::Storage>(nbytes_, device);
  initialize_autograd();
  validate_view_bounds();
}

Tensor::Tensor(shared_ptr<detail::Storage> storage, DType dtype, Shape shape,
               vector<Shape::stride_type> strides, size_t storage_offset)
    : storage_{std::move(storage)}, dtype_{dtype}, shape_{std::move(shape)},
      strides_{std::move(strides)}, storage_offset_{storage_offset},
      nbytes_{checked_nbytes(shape_.numel(), dtype_)} {
  initialize_autograd();
  validate_view_bounds();
}

Tensor::Tensor(Tensor&& other) noexcept
    : storage_{std::move(other.storage_)}, dtype_{other.dtype_}, shape_{std::move(other.shape_)},
      strides_{std::move(other.strides_)}, storage_offset_{other.storage_offset_},
      nbytes_{other.nbytes_}, autograd_{std::move(other.autograd_)} {
  other.reset_to_empty();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    storage_ = std::move(other.storage_);
    dtype_ = other.dtype_;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    storage_offset_ = other.storage_offset_;
    nbytes_ = other.nbytes_;
    autograd_ = std::move(other.autograd_);
    other.reset_to_empty();
  }
  return *this;
}

Tensor::~Tensor() = default;

void Tensor::reset_to_empty() noexcept {
  storage_.reset();
  strides_.clear();
  storage_offset_ = 0;
  nbytes_ = 0;
  autograd_.reset();
}

void Tensor::swap(Tensor& other) noexcept {
  using std::swap;
  swap(storage_, other.storage_);
  swap(dtype_, other.dtype_);
  swap(shape_, other.shape_);
  swap(strides_, other.strides_);
  swap(storage_offset_, other.storage_offset_);
  swap(nbytes_, other.nbytes_);
  swap(autograd_, other.autograd_);
}

size_t Tensor::rank() const noexcept {
  return shape_.rank();
}

const Shape& Tensor::shape() const noexcept {
  return shape_;
}

span<const Shape::stride_type> Tensor::strides() const noexcept {
  return strides_;
}

size_t Tensor::numel() const noexcept {
  return shape_.numel();
}

DType Tensor::dtype() const noexcept {
  return dtype_;
}

Device Tensor::device() const noexcept {
  return storage_ == nullptr ? Device::cpu() : storage_->device();
}

size_t Tensor::nbytes() const noexcept {
  return nbytes_;
}

bool Tensor::is_contiguous() const noexcept {
  if (numel() == 0 || rank() == 0) {
    return true;
  }

  size_t expected_stride{1};
  for (size_t index{rank()}; index > 0; --index) {
    const size_t axis{index - 1};
    const auto extent{static_cast<size_t>(shape_[axis])};
    if (extent > 1 && strides_[axis] != expected_stride) {
      return false;
    }
    expected_stride *= extent;
  }
  return true;
}

Tensor Tensor::materialize_contiguous() const {
  Tensor output{shape_, dtype_, device()};
  const auto copy_values = [this, &output]<typename T> {
    const auto source_base{reinterpret_cast<const T*>(host_data())};
    auto destination_values{output.span<T>()};
    for (size_t index{0}; index < destination_values.size(); ++index) {
      destination_values[index] = source_base[logical_storage_index(index)];
    }
  };
  switch (dtype_) {
  case DType::Float32:
    copy_values.template operator()<float>();
    break;
  case DType::Float64:
    copy_values.template operator()<double>();
    break;
  case DType::Int32:
    copy_values.template operator()<int32_t>();
    break;
  case DType::Int64:
    copy_values.template operator()<int64_t>();
    break;
  }
  return output;
}

Tensor Tensor::clone() const {
  auto output{materialize_contiguous()};
  if (requires_grad()) {
    detail::record_operation(output, {*this},
                             [](const Tensor& gradient) { return vector<Tensor>{gradient}; });
  }
  return output;
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) {
    return *this;
  }
  auto output{materialize_contiguous()};
  if (requires_grad()) {
    detail::record_operation(output, {*this},
                             [](const Tensor& gradient) { return vector<Tensor>{gradient}; });
  }
  return output;
}

Tensor Tensor::reshape(Shape new_shape) const {
  if (!is_contiguous()) {
    throw invalid_argument{"reshape requires a contiguous tensor"};
  }
  if (new_shape.numel() != numel()) {
    throw invalid_argument{"reshape requires the same number of elements"};
  }
  auto new_strides{new_shape.contiguous_strides()};
  Tensor output{storage_, dtype_, std::move(new_shape), std::move(new_strides), storage_offset_};
  if (requires_grad()) {
    const Shape original_shape{shape_};
    detail::record_operation(output, {*this}, [original_shape](const Tensor& gradient) {
      return vector<Tensor>{gradient.reshape(original_shape)};
    });
  }
  return output;
}

Tensor Tensor::transpose(size_t axis_a, size_t axis_b) const {
  if (axis_a >= rank() || axis_b >= rank()) {
    throw out_of_range{"transpose axis is out of range"};
  }
  vector<Shape::dimension_type> dimensions{shape_.dimensions().begin(), shape_.dimensions().end()};
  auto new_strides{strides_};
  std::swap(dimensions[axis_a], dimensions[axis_b]);
  std::swap(new_strides[axis_a], new_strides[axis_b]);
  Tensor output{storage_, dtype_, Shape{std::move(dimensions)}, std::move(new_strides),
                storage_offset_};
  if (requires_grad()) {
    detail::record_operation(output, {*this}, [axis_a, axis_b](const Tensor& gradient) {
      return vector<Tensor>{gradient.transpose(axis_a, axis_b).contiguous()};
    });
  }
  return output;
}

Tensor Tensor::permute(std::span<const size_t> axes) const {
  if (axes.size() != rank()) {
    throw invalid_argument{"permute requires exactly one entry per tensor axis"};
  }

  vector<bool> seen(rank(), false);
  vector<Shape::dimension_type> dimensions(rank());
  vector<Shape::stride_type> new_strides(rank());
  for (size_t result_axis{0}; result_axis < axes.size(); ++result_axis) {
    const size_t source_axis{axes[result_axis]};
    if (source_axis >= rank()) {
      throw out_of_range{"permute axis is out of range"};
    }
    if (seen[source_axis]) {
      throw invalid_argument{"permute axes must not contain duplicates"};
    }
    seen[source_axis] = true;
    dimensions[result_axis] = shape_[source_axis];
    new_strides[result_axis] = strides_[source_axis];
  }
  vector<size_t> inverse_axes(rank());
  for (size_t result_axis{0}; result_axis < axes.size(); ++result_axis) {
    inverse_axes[axes[result_axis]] = result_axis;
  }

  Tensor output{storage_, dtype_, Shape{std::move(dimensions)}, std::move(new_strides),
                storage_offset_};
  if (requires_grad()) {
    detail::record_operation(output, {*this},
                             [inverse_axes = std::move(inverse_axes)](const Tensor& gradient) {
                               return vector<Tensor>{gradient.permute(inverse_axes).contiguous()};
                             });
  }
  return output;
}

Tensor Tensor::permute(initializer_list<size_t> axes) const {
  return permute(std::span<const size_t>{axes.begin(), axes.size()});
}

Tensor Tensor::expand(Shape target_shape) const {
  if (target_shape.rank() < rank()) {
    throw invalid_argument{"expand target rank must not be smaller than the input rank"};
  }

  vector<Shape::stride_type> expanded_strides(target_shape.rank(), 0);
  const size_t leading_axes{target_shape.rank() - rank()};
  for (size_t source_axis{0}; source_axis < rank(); ++source_axis) {
    const size_t target_axis{leading_axes + source_axis};
    const auto source_extent{shape_[source_axis]};
    const auto target_extent{target_shape[target_axis]};
    if (source_extent == target_extent) {
      expanded_strides[target_axis] = strides_[source_axis];
    } else if (source_extent == 1) {
      expanded_strides[target_axis] = 0;
    } else {
      throw invalid_argument{"expand dimensions are not broadcast-compatible"};
    }
  }

  Tensor output{storage_, dtype_, std::move(target_shape), std::move(expanded_strides),
                storage_offset_};
  if (requires_grad()) {
    const Shape original_shape{shape_};
    detail::record_operation(output, {*this}, [original_shape](const Tensor& gradient) {
      return vector<Tensor>{detail::reduce_gradient_to_shape(gradient, original_shape)};
    });
  }
  return output;
}

Tensor Tensor::detach() const {
  return Tensor{storage_, dtype_, shape_, strides_, storage_offset_};
}

size_t Tensor::checked_nbytes(size_t numel, DType dtype) {
  const auto element_size{size_of(dtype)};
  if (numel != 0 && element_size > numeric_limits<size_t>::max() / numel) {
    throw overflow_error{"Tensor byte size overflow"};
  }
  return numel * element_size;
}

void Tensor::validate_view_bounds() const {
  if (strides_.size() != rank()) {
    throw logic_error{"Tensor stride rank does not match shape rank"};
  }

  const size_t element_size{size_of(dtype_)};
  const size_t storage_elements{storage_ == nullptr ? 0 : storage_->nbytes() / element_size};
  if (numel() == 0) {
    if (storage_offset_ > storage_elements) {
      throw logic_error{"Empty tensor view offset exceeds Storage bounds"};
    }
    return;
  }
  if (storage_ == nullptr) {
    throw logic_error{"Non-empty tensor view has no Storage"};
  }

  size_t maximum_offset{storage_offset_};
  for (size_t axis{0}; axis < rank(); ++axis) {
    const auto extent{static_cast<size_t>(shape_[axis])};
    const size_t steps{extent - 1};
    if (steps != 0 && strides_[axis] > numeric_limits<size_t>::max() / steps) {
      throw overflow_error{"Tensor view offset calculation overflow"};
    }
    const size_t contribution{steps * strides_[axis]};
    if (maximum_offset > numeric_limits<size_t>::max() - contribution) {
      throw overflow_error{"Tensor view offset calculation overflow"};
    }
    maximum_offset += contribution;
  }
  if (maximum_offset >= storage_elements) {
    throw logic_error{"Tensor view exceeds Storage bounds"};
  }
}

size_t Tensor::logical_storage_index(size_t logical_index) const {
  size_t storage_index{storage_offset_};
  for (size_t index{rank()}; index > 0; --index) {
    const size_t axis{index - 1};
    const auto extent{static_cast<size_t>(shape_[axis])};
    const size_t coordinate{logical_index % extent};
    logical_index /= extent;
    storage_index += coordinate * strides_[axis];
  }
  return storage_index;
}

byte* Tensor::mutable_host_data() {
  return storage_ == nullptr ? nullptr : storage_->host_data();
}

const byte* Tensor::host_data() const {
  return storage_ == nullptr ? nullptr : storage_->host_data();
}

void swap(Tensor& left, Tensor& right) noexcept {
  left.swap(right);
}

Tensor zeros(Shape shape, DType dtype, Device device) {
  Tensor tensor{std::move(shape), dtype, device};
  switch (dtype) {
  case DType::Float32:
    tensor.fill<float>(0.0F);
    break;
  case DType::Float64:
    tensor.fill<double>(0.0);
    break;
  case DType::Int32:
    tensor.fill<int32_t>(0);
    break;
  case DType::Int64:
    tensor.fill<int64_t>(0);
    break;
  }
  return tensor;
}

Tensor ones(Shape shape, DType dtype, Device device) {
  Tensor tensor{std::move(shape), dtype, device};
  switch (dtype) {
  case DType::Float32:
    tensor.fill<float>(1.0F);
    break;
  case DType::Float64:
    tensor.fill<double>(1.0);
    break;
  case DType::Int32:
    tensor.fill<int32_t>(1);
    break;
  case DType::Int64:
    tensor.fill<int64_t>(1);
    break;
  }
  return tensor;
}

} // namespace spar

namespace spar::detail {

void validate_same_device(const Tensor& left, const Tensor& right, string_view operation) {
  if (left.device() != right.device()) {
    throw invalid_argument{string{operation} + " requires tensors on the same Device"};
  }
}

Shape broadcast_shape(const Shape& left, const Shape& right) {
  const size_t result_rank{std::max(left.rank(), right.rank())};
  vector<Shape::dimension_type> dimensions(result_rank, 1);
  for (size_t offset{0}; offset < result_rank; ++offset) {
    const auto left_extent{offset < left.rank() ? left[left.rank() - 1 - offset]
                                                : Shape::dimension_type{1}};
    const auto right_extent{offset < right.rank() ? right[right.rank() - 1 - offset]
                                                  : Shape::dimension_type{1}};
    Shape::dimension_type result_extent{0};
    if (left_extent == right_extent) {
      result_extent = left_extent;
    } else if (left_extent == 1) {
      result_extent = right_extent;
    } else if (right_extent == 1) {
      result_extent = left_extent;
    } else {
      throw invalid_argument{"Tensor shapes are not broadcast-compatible"};
    }
    dimensions[result_rank - 1 - offset] = result_extent;
  }
  return Shape{std::move(dimensions)};
}

template <typename T>
void reduce_gradient_values(const Tensor& gradient, Tensor& result, const Shape& original_shape) {
  auto result_values{result.span<T>()};
  const auto result_strides{original_shape.contiguous_strides()};
  const size_t leading_axes{gradient.rank() - original_shape.rank()};
  vector<size_t> coordinates(gradient.rank(), 0);

  for (size_t logical_index{0}; logical_index < gradient.numel(); ++logical_index) {
    size_t remaining{logical_index};
    for (size_t index{gradient.rank()}; index > 0; --index) {
      const size_t axis{index - 1};
      const auto extent{static_cast<size_t>(gradient.shape()[axis])};
      coordinates[axis] = remaining % extent;
      remaining /= extent;
    }

    size_t result_index{0};
    for (size_t axis{0}; axis < original_shape.rank(); ++axis) {
      const size_t coordinate{original_shape[axis] == 1 ? 0 : coordinates[leading_axes + axis]};
      result_index += coordinate * result_strides[axis];
    }
    result_values[result_index] += logical_value<T>(gradient, logical_index);
  }
}

Tensor reduce_gradient_to_shape(const Tensor& gradient, const Shape& original_shape) {
  if (gradient.dtype() != DType::Float32 && gradient.dtype() != DType::Float64) {
    throw logic_error{"Broadcast gradient reduction requires a floating dtype"};
  }
  if (gradient.shape() != broadcast_shape(gradient.shape(), original_shape)) {
    throw logic_error{"Gradient shape cannot be reduced to the requested broadcast parent shape"};
  }

  Tensor result{zeros(original_shape, gradient.dtype(), gradient.device())};
  if (gradient.dtype() == DType::Float32) {
    reduce_gradient_values<float>(gradient, result, original_shape);
  } else {
    reduce_gradient_values<double>(gradient, result, original_shape);
  }
  return result;
}

} // namespace spar::detail
