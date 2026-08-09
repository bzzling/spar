export module spar.loss.cross_entropy;

import std;
export import spar.tensor;

export namespace spar::loss {

enum class Reduction { None, Sum, Mean };

[[nodiscard]] Tensor cross_entropy(const Tensor& logits, const Tensor& targets,
                                   Reduction reduction = Reduction::Mean);

[[nodiscard]] Tensor language_model_cross_entropy(const Tensor& logits, const Tensor& token_ids,
                                                  Reduction reduction = Reduction::Mean);

} // namespace spar::loss
