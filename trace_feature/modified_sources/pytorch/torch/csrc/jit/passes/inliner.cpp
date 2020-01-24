#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

std::string classHierarchy(const Node* node){
  Node* it = node->input(0)->node();
  std::string accessorPath = "__module";

  while(it->inputs().size() > 0){
    accessorPath+=".";
    accessorPath+=it->s(attr::name);
    it=it->input(0)->node();
  }
    return accessorPath;
  }

void inlineCalls(Block* block, int depth = 1000,const std::set<std::string>& basicBlocks=std::set<std::string>()) {
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
            *fun_type->function()->optimized_graph(depth - 1,basicBlocks));
        inlineCallTo(cur, fun_type->function(),depth-1,basicBlocks);
      } break;
      case prim::CallMethod: {

        const std::string& name = cur->s(attr::name);
        if(basicBlocks.count(classHierarchy(cur)) == 0) {
          if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
            auto function = class_type->getMethod(name);
            GRAPH_UPDATE("Inlining method '", function->name(), "' to ", *cur);
            GRAPH_UPDATE(
                "Function body: ", *function->optimized_graph(depth - 1,basicBlocks));
            inlineCallTo(cur, function, depth - 1,basicBlocks);
          }
        }
      } break;
      default: {
        for (auto b : cur->blocks()) {
          inlineCalls(b, depth - 1,basicBlocks);
        }
      } break;
    }
  }
}

void Inline(Graph& graph, int depth,const std::set<std::string>& basicBlocks) {
  GRAPH_DUMP("Before Inlining: ", &graph);
  inlineCalls(graph.block(), depth,basicBlocks);
  GRAPH_DUMP("After Inlining: ", &graph);
}


} // namespace jit
} // namespace torch
