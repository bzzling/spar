export module spar.storage;

import std;
import spar.device;

export namespace spar::detail {

/// Internal reference-counted allocation payload used by Tensor handles.
/// This type is not re-exported by the ordinary `spar` umbrella module.
class Storage final {
public:
  Storage(std::size_t nbytes, Device device);
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;
  ~Storage() noexcept;

  [[nodiscard]] std::byte* host_data();
  [[nodiscard]] const std::byte* host_data() const;
  [[nodiscard]] void* cuda_data();
  [[nodiscard]] const void* cuda_data() const;
  [[nodiscard]] Device device() const noexcept;
  [[nodiscard]] std::size_t nbytes() const noexcept;

private:
  std::size_t nbytes_;
  Device device_;
  std::unique_ptr<std::byte[]> host_data_;
  void* cuda_data_;
};

} // namespace spar::detail
