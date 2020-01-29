# insights
* I've figured out how inline works :)
  after a trace is obtained all the stack info is still available.\
  calling _jit_pass_inline(graph) will populate all stack trace info\
  while inlining we recursively inline all function calls\
  so we have the following:\
    torch._C._jit_pass_inline->Inline->inlineCalls->inlineCallTo->optimized_graph->preoptimizeGraph->Inline\
    we can add depth/basic blocks parameters which will have default non influencing values thus we can have traces and profiles at the same resolution without information loss

DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_CUDA=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 python setup.py

## instructions:
* run copy_and_build_pytorch.sh
* run feature_test.py to make sure the feature works as expected