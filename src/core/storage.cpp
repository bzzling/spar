module spar.storage;

import std;
import spar.device;

using namespace std;

namespace spar::detail {

Storage::Storage(size_t nbytes, Device device) : nbytes_{nbytes}, device_{device}, data_{} {
  if (!device_.is_cpu()) {
    throw runtime_error{"CUDA storage is not implemented"};
  }
  data_ = nbytes_ == 0 ? nullptr : make_unique_for_overwrite<byte[]>(nbytes_);
}

Storage::~Storage() = default;

byte* Storage::host_data() {
  if (!device_.is_cpu()) {
    throw logic_error{"Storage is not host-accessible"};
  }
  return data_.get();
}

const byte* Storage::host_data() const {
  if (!device_.is_cpu()) {
    throw logic_error{"Storage is not host-accessible"};
  }
  return data_.get();
}

Device Storage::device() const noexcept {
  return device_;
}

size_t Storage::nbytes() const noexcept {
  return nbytes_;
}

} // namespace spar::detail
