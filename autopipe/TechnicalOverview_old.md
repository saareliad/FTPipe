# Technical Overview

This repo Mixed-pipe partitioning of PyTorch models while ensuring transparency with the original model.

# High Level API

```python
def pipe_model(model, ...)
```

this function does partitioning end to end using profiling and code generation
returns a partitioned graph object.

### Computational Graph Representation

[model_profiling/control_flow_graph.py](autopipe/model_profiling/control_flow_graph.py)



# Model Profiling

## Model Profiler

this module is responsible to gather various metrics about all layers (and optionally operations) of the model

- forward_time in ms
- backward_time ms
- cuda_memory_forward MB
- cuda_memory_backward MB
- size (i.e parameters, buffers) MB
- input_size MB
- output_size MB

each block runs in isolation.

the main parameters that influence such profiling are depth and basic blocks:

- depth dictates how far down to go in the model's class hierarchy
- basic blocks enable us a fine grained control about which layers to profile as a whole

# Limitations

as we are using tracing there are several limitations that come with it:

- only tensors and nested lists/tuples of tensors are supported as model inputs
- control flow must be deterministic. we can only profile the actions that were taken for the traced input.\
  for example if statement will be inlined with the path taken same for loops which will be unrolled. Limited amount of eager execution is allowed inside `basic_blocks`.

### Pytorch tracing problems (deprecated)
- as pytorch has problem with tracing keyword given to forward methods, it is advised to pass keywords by positions.\
  for example:

  ```python
  def forward(x,y=None,z=None)...

  model(x,z=10)
  ```

  the following will not register z as the third input but as the second, so if you know that only z will be used rewrite it to be:

  ```python
  def forward(x,z=None,y=None)
  ```

  that will ensure that the generated code is correct.
  similarly if we do:

  ```python
  model(x,z=x)
  ```

  we can't know that x was passed twice, instead it will appear as `model(x)`

- functions that have string args like for example nll_loss which has a reduction arg will not register correctly and must be fixed manually
- trace bugs
  - nested tuples are flattened
  - unpack uses not used in correct locations opened an issue


# Model Partitioning
Mixed Pipe is activated automatically in everything except "acyclic".
 * METIS partKway
 * mpipe - Mixed-Pipe
 * acyclic (and multililevel)


# Code Generation
The compiler module is responsible for code generation.
## Partitioned model File

This module is the gateway to generating partitioned code.
it's responsible to generate helper methods to be used by the generated code.


## Supported Pytorch Functions

mostly everything

- torch
- Tensor
- functional


# Environment
See [environment.yml](environment.yml)

# TODOS
- integration
- lstms and packed_sequences are problematic
- string arguments for functions like nll_loss are not supported with tracing but yes with scripting
- optional inputs problem
