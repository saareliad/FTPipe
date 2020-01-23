# instructions
copy contents of modified/csrc into torch/csrc


# modifications 
file
lineNumberOfOriginalFile => new line

- torch/csrc/jit/passes/inliner.h
    - line 9 => TORCH_API void Inline(Graph& graph,int depth=1000);

- torch/csrc/jit/passes/inliner.cpp
  - line 13 => 
        void inlineCalls(Block* block, int depth) {
            if (depth == 0) {
                return;
            }
  - lines 29-31 => 
        GRAPH_UPDATE(
            "Function body: ", *fun_type->function()->optimized_graph(depth-1));
        inlineCallTo(cur, fun_type->function(),depth-1);
  - lines 38-39 =>
        GRAPH_UPDATE("Function body: ", *function->optimized_graph(depth-1));
        inlineCallTo(cur, function,depth-1);
  - line 44 => inlineCalls(b,depth-1);
  - line 48 => void Inline(Graph& graph,int depth) {
  
- torch/csrc/jit/ir.h
  - line 1352 => TORCH_API std::vector<Value*> inlineCallTo(Node* to_replace, Function* callee,int depth=1000);

- torch/csrc/jit/ir.cpp
  - line 1720 => std::vector<Value*> inlineCallTo(Node* to_replace, Function* callee,int depth) {
  - line 1723 =>   
    auto new_outputs = insertGraph(*to_replace->owningGraph(),*(callee->optimized_graph(depth)),to_replace->inputs(),value_map);

- torch/csrc/jit/function.h
  - line 12 => TORCH_API void preoptimizeGraph(std::shared_ptr<Graph>& graph,int depth=1000);
  - line 37 => std::shared_ptr<Graph> optimized_graph(int depth=1000) const {
  - line 43 => preoptimizeGraph(*optimized_graph_,depth);

- torch/csrc/jit/function.cpp
  - line 71 => void preoptimizeGraph(std::shared_ptr<Graph>& graph, int depth) {
  - line 74 => Inline(*graph, depth);