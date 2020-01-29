#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}


//terminal layer is a layer that does not contain other layers
bool isTerminalLayer(const Function* function) {
  for (const Node* it : function->graph()->nodes()) {
    if (it->kind() == prim::CallMethod) {
      return false;
    }
  }
  return true;
}

//we exand layers if they are not basic blocks and not terminal
bool canExpandLayerCall(const Function* layerCall,int depth,const std::set<std::string>& basicBlocks,const std::string& accessorPath){
  return (basicBlocks.count(accessorPath) == 0) && (!isTerminalLayer(layerCall));
}

void inlineCalls(Block* block, int depth = -1,const std::set<std::string>& basicBlocks=std::set<std::string>(),const std::string& accessorPath="") {
  if (depth == 0){
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
            *fun_type->function()->optimized_graph(depth - 1,basicBlocks,accessorPath));
        inlineCallTo(cur, fun_type->function(),depth - 1,basicBlocks,accessorPath);
      } break;
      case prim::CallMethod: {
        const std::string& name = cur->s(attr::name);
        if (auto class_type = cur->input(0)->type()->cast<ClassType>()) {
          const std::string& accessorName =cur->input(0)->node()->s(attr::name);
          auto function = class_type->getMethod(name);
          
          //we inline if:
          //  layer is not a basic block
          //  layer is composite and depth > 0
          //  depth < 0 indicates that it was not us that called this method so we do not interfere
          if (depth < 0 || canExpandLayerCall(function,depth,basicBlocks,accessorPath+"."+accessorName)) {
            GRAPH_UPDATE("Inlining method '", function->name(), "' to ", *cur);
            GRAPH_UPDATE(
                "Function body: ", *function->optimized_graph(depth - 1,basicBlocks,accessorPath+"."+accessorName));
            inlineCallTo(cur, function, depth - 1,basicBlocks,accessorPath+"."+accessorName);
          }
        }
      } break;
      default: {
        for (auto b : cur->blocks()) {
          inlineCalls(b, depth - 1,basicBlocks,accessorPath);
        }
      } break;
    }
  }
}

void Inline(Graph& graph, int depth,const std::set<std::string>& basicBlocks,const std::string& accessorPath) {
  GRAPH_DUMP("Before Inlining: ", &graph);
  inlineCalls(graph.block(), depth,basicBlocks,accessorPath);
  GRAPH_DUMP("After Inlining: ", &graph);
}


} // namespace jit
} // namespace torch
