#pragma once

#include <torch/csrc/jit/ir.h>
#include<string>
#include<set>

namespace torch {
namespace jit {

// Inline function and method calls.
TORCH_API void Inline(Graph& graph, int depth = 1000,const std::set<std::string>& basicBlocks = std::set<std::string>());

} // namespace jit
} // namespace torch
