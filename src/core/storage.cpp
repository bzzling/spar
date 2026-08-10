module spar.storage;

import std;
import spar.cuda_runtime;
import spar.device;

using namespace std;

namespace spar::detail {

Storage::Storage(size_t nbytes, Device device)
    : nbytes_{nbytes}, device_{device}, host_data_{}, cuda_data_{nullptr} {
  if (device_.is_cpu()) {
    host_data_ = nbytes_ == 0 ? nullptr : make_unique_for_overwrite<byte[]>(nbytes_);
    return;
  }
  cuda_data_ = cuda::allocate(nbytes_, device_.index());
}

Storage::~Storage() noexcept {
  if (device_.is_cuda()) {
    cuda::deallocate(cuda_data_, device_.index());
  }
}

byte* Storage::host_data() {
  if (!device_.is_cpu()) {
    throw logic_error{"Storage is not host-accessible"};
  }
  return host_data_.get();
}

const byte* Storage::host_data() const {
  if (!device_.is_cpu()) {
    throw logic_error{"Storage is not host-accessible"};
  }
  return host_data_.get();
}

void* Storage::cuda_data() {
  if (!device_.is_cuda()) {
    throw logic_error{"Storage is not CUDA-accessible"};
  }
  return cuda_data_;
}

const void* Storage::cuda_data() const {
  if (!device_.is_cuda()) {
    throw logic_error{"Storage is not CUDA-accessible"};
  }
  return cuda_data_;
}

Device Storage::device() const noexcept {
  return device_;
}

size_t Storage::nbytes() const noexcept {
  return nbytes_;
}

} // namespace spar::detail
