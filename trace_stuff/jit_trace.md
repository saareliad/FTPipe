# insights

- there is a flag indicating whether to inline or not default is False:\
  declared here torch/csrc/jit/script/module.cpp\
  can be set with `torch._C._jit_set_inline_everything_mode(True)`

* in pytorch/torch/csrc/jit/tracer.cpp in the function trace we can see that there is a check wether to inline the graph\
  we can insert logic there that can be based on depth or traced module instance

## TODOS

- how tracing goes all the way down? is it a global flag or we call it recursively somehow?
- how to pass a depth variable?
  - maybe a global variable?(what if multiple threads)
  - if we gain insight to first point we can solve this

* how to handle basic blocks?

  - same as depth and we can use a dynamic class,\
    or std::is_base_of to check if a basic block

* if we inline some calls and not others than will tuple/list packing/ unpacking will be handeled correctly?
