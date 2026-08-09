module spar.nn.functional;

import spar.ops.elementwise;
import spar.ops.unary;
import spar.tensor;

namespace spar::nn {

Tensor swiglu(const Tensor& gate, const Tensor& value) {
  return multiply(silu(gate), value);
}

} // namespace spar::nn
