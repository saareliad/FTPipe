# insights
* I've figured out how inline works :)
  after a trace is obtained all the stack info is still available.\
  calling _jit_pass_inline(graph) will populate all stack trace info\
  while inlining we recursively inline all function calls\
  so we have the following:\
    torch._C._jit_pass_inline->Inline->inlineCalls->inlineCallTo->optimized_graph->preoptimizeGraph->Inline\
    we can add depth/basic blocks parameters which will have default non influencing values thus we can have traces and profiles at the same resolution without information loss

## TODOS
* implement solution for depth dependent tracing
* how to handle basic blocks?
  - need to be able to access the nodes class during the trace should be possible via the prim::GetAttr Node
  - same as depth and we can use a dynamic cast
    or std::is_base_of to check if a basic block

* if we inline some calls and not others than will tuple/list packing/ unpacking will be handeled correctly?




## 