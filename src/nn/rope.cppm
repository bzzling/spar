export module spar.nn.rope;

import std;
export import spar.tensor;

export namespace spar::nn {

/// Applies Spar's interleaved-pair RoPE convention to [..., T, D].
/// Pair i is (2i, 2i+1), with frequency theta^(-2i/D).
[[nodiscard]] Tensor apply_rope(const Tensor& input, std::size_t start_position = 0,
                                double theta = 10000.0);

} // namespace spar::nn
