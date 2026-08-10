export module spar.device;

import std;

export namespace spar {

enum class DeviceType : std::uint8_t {
  CPU,
  CUDA,
};

/// Runtime tensor placement. CUDA values describe a future placement target; CUDA allocation is
/// deliberately unsupported until Spar has a real CUDA backend.
class Device final {
public:
  [[nodiscard]] static constexpr Device cpu() noexcept {
    return Device{DeviceType::CPU, 0};
  }

  [[nodiscard]] static Device cuda(std::int32_t index);

  [[nodiscard]] constexpr DeviceType type() const noexcept {
    return type_;
  }

  [[nodiscard]] constexpr std::int32_t index() const noexcept {
    return index_;
  }

  [[nodiscard]] constexpr bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }

  [[nodiscard]] constexpr bool is_cuda() const noexcept {
    return type_ == DeviceType::CUDA;
  }

  friend constexpr bool operator==(const Device&, const Device&) noexcept = default;

private:
  constexpr Device(DeviceType type, std::int32_t index) noexcept : type_{type}, index_{index} {}

  DeviceType type_;
  std::int32_t index_;
};

} // namespace spar
