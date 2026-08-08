export module spar.dtype;

import std;

export namespace spar {

/// runtime element types supported by Spar tensors.
enum class DType : std::uint8_t {
  Float32,
  Float64,
  Int32,
  Int64,
};

/// returns the number of storage bytes used by one value of `dtype`.
[[nodiscard]] constexpr std::size_t size_of(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return sizeof(float);
  case DType::Float64:
    return sizeof(double);
  case DType::Int32:
    return sizeof(std::int32_t);
  case DType::Int64:
    return sizeof(std::int64_t);
  }
  throw std::invalid_argument{"Unknown Spar dtype"};
}

/// returns the stable lowercase name of `dtype`.
[[nodiscard]] constexpr std::string_view name_of(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return "float32";
  case DType::Float64:
    return "float64";
  case DType::Int32:
    return "int32";
  case DType::Int64:
    return "int64";
  }
  throw std::invalid_argument{"Unknown Spar dtype"};
}

/// maps a supported C++ scalar type to its Spar runtime dtype.
template <typename T> [[nodiscard]] consteval DType dtype_of() {
  using Value = std::remove_cv_t<T>;
  if constexpr (std::is_same_v<Value, float>) {
    return DType::Float32;
  } else if constexpr (std::is_same_v<Value, double>) {
    return DType::Float64;
  } else if constexpr (std::is_same_v<Value, std::int32_t>) {
    return DType::Int32;
  } else if constexpr (std::is_same_v<Value, std::int64_t>) {
    return DType::Int64;
  } else {
    static_assert(!sizeof(T), "This C++ type has no Spar dtype mapping");
  }
}

} // namespace spar
