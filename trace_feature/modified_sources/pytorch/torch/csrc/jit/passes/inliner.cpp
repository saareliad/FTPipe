#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

void inlineCalls(Block* block, int depth = 1000) {
  if (depth == 0) {
    return;
  }
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* cur = *it++;
    switch (cur->kind()) {
      case prim::CallFunction: {
        AT_ASSERT(cur->input(0)->node()->kind() == prim::Constant);
        auto function_constant = cur->input(0)->node();
        auto fun_type =
            function_constant->output()->type()->expect<FunctionType>();
        cur->removeInput(0);
        GRAPH_UPDATE(
            "Inlining function '", fun_type->function()->name(), "' to ", *cur);
        GRAPH_UPDATE(
            "Function body: ",
            *fun_type->function()->optimized_graph(depth - 1));
        inlineCallTo(cur, fun_type->function(),depth-1);
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
          auto function = class_type->getMethod(name);
          GRAPH_UPDATE("Inlining method '", function->name(), "' to ", *cur);
          GRAPH_UPDATE(
              "Function body: ", *function->optimized_graph(depth - 1));
          inlineCallTo(cur, function, depth - 1);
        }
      } break;
      default: {
        for (auto b : cur->blocks()) {
          inlineCalls(b, depth - 1);
        }
      } break;
    }
  }
}

void Inline(Graph& graph, int depth) {
  GRAPH_DUMP("Before Inlining: ", &graph);
  inlineCalls(graph.block(), depth);
  GRAPH_DUMP("After Inlining: ", &graph);
}

} // namespace jit
} // namespace torch
