module spar.tensor;

import std;
import spar.dtype;
import spar.shape;

using namespace std;

namespace spar::detail {

struct AutogradNode final {
  vector<Tensor> parents;
  BackwardFunction backward;
};

struct AutogradMeta final {
  bool requires_grad{false};
  shared_ptr<AutogradNode> grad_fn;
  optional<Tensor> leaf_grad;
};

struct AutogradAccess final {
  [[nodiscard]] static shared_ptr<AutogradMeta>& meta(Tensor& tensor) {
    return tensor.autograd_;
  }

  [[nodiscard]] static const shared_ptr<AutogradMeta>& meta(const Tensor& tensor) {
    return tensor.autograd_;
  }
};

void record_operation(Tensor& output, vector<Tensor> requiring_parents, BackwardFunction backward) {
  if (requiring_parents.empty()) {
    throw logic_error{"Cannot record an autograd operation without requiring-grad parents"};
  }
  for (const Tensor& parent : requiring_parents) {
    if (!parent.requires_grad()) {
      throw logic_error{"Autograd nodes may retain only requiring-grad parents"};
    }
  }

  auto& output_meta{AutogradAccess::meta(output)};
  output_meta = make_shared<AutogradMeta>();
  output_meta->requires_grad = true;
  output_meta->grad_fn =
      make_shared<AutogradNode>(AutogradNode{std::move(requiring_parents), std::move(backward)});
}

} // namespace spar::detail

namespace spar {
namespace {

void add_gradient_values(Tensor& destination, const Tensor& contribution) {
  if (destination.shape() != contribution.shape() || destination.dtype() != contribution.dtype()) {
    throw logic_error{"Internal gradient accumulation shape or dtype mismatch"};
  }
  if (!destination.is_contiguous() || !contribution.is_contiguous()) {
    throw logic_error{"Internal gradient accumulation requires contiguous tensors"};
  }

  const auto add_values = [&destination, &contribution]<typename T> {
    auto destination_values{destination.span<T>()};
    const auto contribution_values{contribution.span<T>()};
    for (size_t index{0}; index < destination_values.size(); ++index) {
      destination_values[index] += contribution_values[index];
    }
  };

  switch (destination.dtype()) {
  case DType::Float32:
    add_values.template operator()<float>();
    break;
  case DType::Float64:
    add_values.template operator()<double>();
    break;
  case DType::Int32:
  case DType::Int64:
    throw logic_error{"Integer gradients are not supported"};
  }
}

[[nodiscard]] Tensor independent_gradient(const Tensor& contribution, const Tensor& parent) {
  if (contribution.shape() != parent.shape() || contribution.dtype() != parent.dtype()) {
    throw logic_error{"Backward rule produced a gradient with incorrect shape or dtype"};
  }
  if (contribution.requires_grad()) {
    throw logic_error{"Backward rule produced a higher-order gradient"};
  }
  return contribution.detach().clone();
}

} // namespace

void Tensor::initialize_autograd() {
  autograd_ = make_shared<detail::AutogradMeta>();
}

bool Tensor::requires_grad() const noexcept {
  return autograd_ != nullptr && autograd_->requires_grad;
}

bool Tensor::is_leaf() const noexcept {
  return autograd_ == nullptr || autograd_->grad_fn == nullptr;
}

bool Tensor::has_grad() const noexcept {
  return autograd_ != nullptr && autograd_->leaf_grad.has_value();
}

void Tensor::set_requires_grad(bool enabled) {
  if (!is_leaf()) {
    throw invalid_argument{"requires_grad may only be changed on leaf tensors"};
  }
  if (enabled && dtype_ != DType::Float32 && dtype_ != DType::Float64) {
    throw invalid_argument{"Only Float32 and Float64 tensors may require gradients"};
  }
  if (autograd_ == nullptr) {
    initialize_autograd();
  }
  autograd_->requires_grad = enabled;
  if (!enabled) {
    autograd_->leaf_grad.reset();
  }
}

Tensor Tensor::grad() const {
  if (!is_leaf()) {
    throw invalid_argument{"Gradients are retained for leaf tensors only"};
  }
  if (!has_grad()) {
    throw logic_error{"Tensor has no accumulated gradient"};
  }
  return *autograd_->leaf_grad;
}

void Tensor::zero_grad() {
  if (!is_leaf()) {
    throw invalid_argument{"zero_grad is supported for leaf tensors only"};
  }
  if (autograd_ != nullptr) {
    autograd_->leaf_grad.reset();
  }
}

void Tensor::backward() {
  if (!requires_grad()) {
    throw invalid_argument{"backward requires a Tensor that tracks gradients"};
  }
  if (numel() != 1) {
    throw invalid_argument{"backward without an explicit gradient requires a one-element Tensor"};
  }
  if (dtype_ != DType::Float32 && dtype_ != DType::Float64) {
    throw invalid_argument{"backward supports Float32 and Float64 losses only"};
  }

  vector<shared_ptr<detail::AutogradMeta>> topological_order;
  unordered_set<detail::AutogradMeta*> visited;
  const auto visit = [&topological_order, &visited](
                         const auto& self, const shared_ptr<detail::AutogradMeta>& meta) -> void {
    if (meta == nullptr || !meta->requires_grad || !visited.insert(meta.get()).second) {
      return;
    }
    if (meta->grad_fn != nullptr) {
      for (const Tensor& parent : meta->grad_fn->parents) {
        self(self, parent.autograd_);
      }
    }
    topological_order.push_back(meta);
  };
  visit(visit, autograd_);

  unordered_map<detail::AutogradMeta*, Tensor> gradients;
  gradients.emplace(autograd_.get(), ones(shape_, dtype_));

  for (auto iterator{topological_order.rbegin()}; iterator != topological_order.rend();
       ++iterator) {
    const auto& meta{*iterator};
    auto gradient_iterator{gradients.find(meta.get())};
    if (gradient_iterator == gradients.end()) {
      throw logic_error{"Reachable autograd identity did not receive a gradient"};
    }
    Tensor& gradient{gradient_iterator->second};

    if (meta->grad_fn == nullptr) {
      if (meta->leaf_grad.has_value()) {
        add_gradient_values(*meta->leaf_grad, gradient);
      } else {
        meta->leaf_grad = gradient.clone();
      }
      continue;
    }

    const auto contributions{meta->grad_fn->backward(gradient)};
    if (contributions.size() != meta->grad_fn->parents.size()) {
      throw logic_error{"Backward rule returned an incorrect number of gradients"};
    }

    for (size_t index{0}; index < contributions.size(); ++index) {
      const Tensor& parent{meta->grad_fn->parents[index]};
      Tensor contribution{independent_gradient(contributions[index], parent)};
      auto existing{gradients.find(parent.autograd_.get())};
      if (existing == gradients.end()) {
        gradients.emplace(parent.autograd_.get(), std::move(contribution));
      } else {
        add_gradient_values(existing->second, contribution);
      }
    }
  }
}

} // namespace spar
