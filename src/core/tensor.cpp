module spar.tensor;

import std;
import spar.dtype;
import spar.shape;

using namespace std;

namespace spar {

Tensor::Tensor(Shape shape, DType dtype)
    : dtype_{dtype}, shape_{std::move(shape)}, strides_{shape_.contiguous_strides()},
      nbytes_{checked_nbytes(shape_.numel(), dtype_)}, storage_{allocate(nbytes_)} {}

Tensor::Tensor(const Tensor& other)
    : dtype_{other.dtype_}, shape_{other.shape_}, strides_{other.strides_}, nbytes_{other.nbytes_},
      storage_{allocate(nbytes_)} {
  if (nbytes_ != 0) {
    memcpy(storage_.get(), other.storage_.get(), nbytes_);
  }
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    Tensor copy{other};
    swap(copy);
  }
  return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : dtype_{other.dtype_}, shape_{std::move(other.shape_)}, strides_{std::move(other.strides_)},
      nbytes_{other.nbytes_}, storage_{std::move(other.storage_)} {
  other.reset_to_empty();
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    dtype_ = other.dtype_;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    nbytes_ = other.nbytes_;
    storage_ = std::move(other.storage_);
    other.reset_to_empty();
  }
  return *this;
}

Tensor::~Tensor() = default;

void Tensor::reset_to_empty() noexcept {
  strides_.clear();
  nbytes_ = 0;
  storage_.reset();
}

void Tensor::swap(Tensor& other) noexcept {
  using std::swap;
  swap(dtype_, other.dtype_);
  swap(shape_, other.shape_);
  swap(strides_, other.strides_);
  swap(nbytes_, other.nbytes_);
  swap(storage_, other.storage_);
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
size_t Tensor::nbytes() const noexcept {
  return nbytes_;
}

size_t Tensor::checked_nbytes(size_t numel, DType dtype) {
  const auto element_size{size_of(dtype)};
  if (numel != 0 && element_size > numeric_limits<size_t>::max() / numel) {
    throw overflow_error{"Tensor byte size overflow"};
  }
  return numel * element_size;
}

unique_ptr<byte[]> Tensor::allocate(size_t nbytes) {
  if (nbytes == 0) {
    return {};
  }
  return make_unique_for_overwrite<byte[]>(nbytes);
}

void swap(Tensor& left, Tensor& right) noexcept {
  left.swap(right);
}

Tensor zeros(Shape shape, DType dtype) {
  Tensor tensor{std::move(shape), dtype};
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

Tensor ones(Shape shape, DType dtype) {
  Tensor tensor{std::move(shape), dtype};
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
