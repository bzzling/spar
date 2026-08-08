export module spar.ops.reduction;

import std;
export import spar.tensor;

export namespace spar {

[[nodiscard]] Tensor sum(const Tensor& input);
[[nodiscard]] Tensor sum(const Tensor& input, std::span<const std::size_t> axes,
                         bool keepdim = false);
[[nodiscard]] Tensor sum(const Tensor& input, std::size_t axis, bool keepdim = false);
[[nodiscard]] Tensor sum(const Tensor& input, std::initializer_list<std::size_t> axes,
                         bool keepdim = false);
[[nodiscard]] Tensor mean(const Tensor& input);
[[nodiscard]] Tensor mean(const Tensor& input, std::span<const std::size_t> axes,
                          bool keepdim = false);
[[nodiscard]] Tensor mean(const Tensor& input, std::size_t axis, bool keepdim = false);
[[nodiscard]] Tensor mean(const Tensor& input, std::initializer_list<std::size_t> axes,
                          bool keepdim = false);
[[nodiscard]] Tensor reduce_max(const Tensor& input);

} // namespace spar
