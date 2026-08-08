export module spar.storage;

import std;

export namespace spar::detail {

/// Internal reference-counted allocation payload used by Tensor handles.
/// This type is not re-exported by the ordinary `spar` umbrella module.
class Storage final {
public:
  explicit Storage(std::size_t nbytes);
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;
  ~Storage();

  [[nodiscard]] std::byte* data() noexcept;
  [[nodiscard]] const std::byte* data() const noexcept;
  [[nodiscard]] std::size_t nbytes() const noexcept;

private:
  std::size_t nbytes_;
  std::unique_ptr<std::byte[]> data_;
};

} // namespace spar::detail
