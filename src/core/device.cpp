module spar.device;

import std;

namespace spar {

Device Device::cuda(std::int32_t index) {
  if (index < 0) {
    throw std::invalid_argument{"CUDA device index must be nonnegative"};
  }
  return Device{DeviceType::CUDA, index};
}

} // namespace spar
