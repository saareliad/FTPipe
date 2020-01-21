#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

// Inline function and method calls.
TORCH_API void Inline(Graph& graph, int depth = 1000);

} // namespace jit
} // namespace torch
