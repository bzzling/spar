module spar.storage;

import std;

using namespace std;

namespace spar::detail {

Storage::Storage(size_t nbytes)
    : nbytes_{nbytes}, data_{nbytes == 0 ? nullptr : make_unique_for_overwrite<byte[]>(nbytes)} {}

Storage::~Storage() = default;

byte* Storage::data() noexcept {
  return data_.get();
}

const byte* Storage::data() const noexcept {
  return data_.get();
}

size_t Storage::nbytes() const noexcept {
  return nbytes_;
}

} // namespace spar::detail
