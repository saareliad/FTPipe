# Technical Overview

This repo can perform dynamic partitioning of PyTorch models while ensuring balance and full transperency with the original model.\
This overview is organized into sections that go over different independent components:

Sections start with a reference to the source file where the code related to the section resides.

## Table of Contents

- [Technical Overview](#technical-overview)
  - [Table of Contents](#table-of-contents)
- [High Level API](#high-level-api)
- [Model Profiling](#model-profiling)
  - [Model Profiler](#model-profiler)
  - [Control Flow Graph](#control-flow-graph)
    - [Graph Builder](#graph-builder)
    - [Graph Pitfalls](#graph-pitfalls)
    - [Graph Representation](#graph-representation)
- [Model Partitioning](#model-partitioning)
  - [METIS Partitioning](#metis-partitioning)
  - [Pipedream Partitioning](#pipedream-partitioning)
  - [Partitioning Post Processing](#partitioning-post-processing)
- [Code Generation](#code-generation)
  - [Partitioned model File](#partitioned-model-file)
  - [Compiler](#compiler)
    - [Constructor Method](#constructor-method)
    - [Forward Method](#forward-method)
    - [State Methods](#state-methods)
  - [Supported Pytorch Functions](#supported-pytorch-functions)
    - [Supported Types](#supported-types)
    - [Declaration Parsing](#declaration-parsing)
- [Synchronous Pipeline](#synchronous-pipeline)
- [Experiments](#experiments)
- [Environment](#environment)
- [TODOS](#todos)

# High Level API

[init.py](pytorch_Gpipe/__init__.py)

our high level API can be described as follows in a hierarchical order

```python
def pipe_model(model, sample_batch, kwargs, n_iter, nparts,
               depth, basic_blocks,partition_by_memory,
               node_weight_function,edge_weight_function,
               output_file, DEBUG, **METIS_opt)
```

this function does partitioning end to end using profiling and code generation
returns a partitioned graph object.

```python
def partition_with_profiler(model, sample_batch, kwargs, n_iter, nparts, max_depth,
                              basic_blocks,node_weight_function,edge_weight_function, **METIS_opt)
```

this function profiles and partitions the model without code generation
returns a partitioned graph object

```python
def METIS_partition(graph,num_partitions,node_weight_function,edge_weight_function,**METIS_opts)
```

this functions performs METIS kway partitioning on our graph
we support custom weighting functions for nodes and edges\
if not specified a default value of 1 will be given

```python
def profile_network(net, sample_batch,kwargs, basic_blocks, max_depth,n_iter)
```

this function performs profiling of the network emiting a Profile object for each layer

```python
def build_graph(model, sample_batch, kwargs, max_depth, basic_blocks, use_profiler, n_iter, weights)
```

this function builds a graph representing the model

```python
def compile_partitoned_model(graph, model, verbose, output_file):
```

this function takes the graph and compiles it emitting python code

# Model Profiling

## Model Profiler

[model_profiling/network_profiler.py](pytorch_Gpipe/model_profiling/network_profiler.py)

this module is responsible to gather various metrics about all layers of the model
emiting a Profile object for each layer:

- forward_time in ms
- backward_time ms
- cuda_memory_forward mb
- cuda_memory_backward mb
- layer_size mb
- input_size mb
- output_size mb

each layer is run in isolation to ensure accurate results.

```python
def profile_network(net, sample_batch,kwargs, basic_blocks, max_depth,n_iter)
```

the main parameters that influence such profiling are depth and basic blocks:

- depth dictates how far down to go in the model's class hierarchy
- basic blocks enable us a fine graid control about which layers to profile as a awhole

## Control Flow Graph

### Graph Builder

[model_profiling/graph_builder.py](pytorch_Gpipe/model_profiling/graph_builder.py)

this module is tasked with converting a model into our graph representation using tracing.
it digests the raw trace and omits a graph detailing the model according to given depth and basic blocks config

```python
def build_graph(model, sample_batch, kwargs, max_depth, basic_blocks, use_profiler, n_iter, weights)
```

### Graph Pitfalls

as we are using tracing there are several limitations that come with it:

- only tensors and nested lists/tuples of tensors are supported as model inputs
- control flow must be deterministic. we can only profile the actions that were taken for the traced input.\
  for example if statement will be inlined with the path taken same for loops which will be unrolled.
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

### Graph Representation

[model_profiling/control_flow_graph.py](pytorch_Gpipe/model_profiling/control_flow_graph.py)

our Graph is a simple data structre that holds nodes and does not have special logic

# Model Partitioning

## METIS Partitioning

[model_partitioning/partition_graph.py](pytorch_Gpipe/model_partitioning/partition_graph.py)

this module involves the necessary calls in order to perform METIS partKway method on our graph

```python
def METIS_partition(graph, num_partitions,node_weight_function,edge_weight_function,**METIS_opts)
```

## Pipedream Partitioning

[model_partitioning/pipedream_partition.py](pytorch_Gpipe/model_partitioning/pipedream_partition.py)
this module involves the necessary logic in order to perform pipedream partition method on our graph
#TODO not yet functional

## Partitioning Post Processing

[model_partitioning/process_partition.py](pytorch_Gpipe/model_partitioning/process_partition.py)

this module takes the partitioned graph and applies various fixes to problem that can occur due to
the partition process

# Code Generation

## Partitioned model File

[compiler/compile_partitioned_model.py](pytorch_Gpipe/compiler/compile_partitioned_model.py)

this module is the gateway to generating partitioned code.
it's responsible to generate helper methods to be used by the generated code.

```python
def compile_partitoned_model(graph, model, verbose, output_file):
```

the verbose argument makes the code generator to emit each statement in forward method in a new line(useful for debuging)

the partitions can be consumed later by calling the generated method:

```python
def create_pipeline_configuration(model,DEBUG,partitions_only)
```

DEBUG is wether we want the partitions on the CPU/GPUS
partitions_only is wether we want oly the partitions or the entire config

the config is composed of a dicionary with the following fields:

- model inputs: the input scopes of the model (one for each tensor)
- model outputs: the output scopes of the model sorted by name
- for each partition from 0 to nParts-1
  - inputs: partition input scopes sorted by name
  - outputs: partition output scopes sorted by name
  - model: the actual partition module

## Compiler

### Constructor Method

[compiler/constructor.py](pytorch_Gpipe/compiler/constructor.py)

generates code for partition **init** method

```python
  def generate_init_method(class_name, full_names, layer_classes,
                        is_param_dict, buff_param_names)
```

### Forward Method

[compiler/partition_forward_method.py](pytorch_Gpipe/compiler/partition_forward_method.py)

this module is tasked with generation the entire forward method for each partition.
code is generated according to topological sorting of the partition

```python
def generate_forward_method(partition, model_outputs,
                            scope_to_class_field, verbose):
```

### State Methods

[compiler/state_methods.py](pytorch_Gpipe/compiler/state_methods.py)

generates the load/state dict methods and the named param/buffer methods
ensuring full transperency between partitioned and original model

```python
def generate_state_methods()
```

## Supported Pytorch Functions

this sections involves logic needed in order to instantiate method calls for pytorch's namespaces:

- torch
- Tensor
- functional

### Supported Types

[compiler/supported_types.py](pytorch_Gpipe/compiler/supported_types.py)

contains declearations for our supported type each type can be then compared against a given type
in order to perform type matching

### Declaration Parsing

[compiler/parse_declarations.py](pytorch_Gpipe/compiler/parse_declarations.py)

contains our logic in order to parse [declerations.py](pytorch_Gpipe/compiler/declarations.txt)
which contains all of the methods we support.

for each functions we create a Function representation which we can then compare againt given types
in order to perfrom overload resolution and generate the correct function call.

# Synchronous Pipeline

[pipeline.py](pytorch_Gpipe/pipeline.py)

contains our logic for synchronous pipeline according to [Gpipe](https://arxiv.org/pdf/1811.06965.pdf)

```python
  class Pipeline()
```

TODO still needs testing tested on cpus and single GPU


# Experiments
 we provide the code we used to run our experiments.
 we've implemented throughput/memory and accuracy experiments

# Environment

- python 3.7
- pytorch 1.4
- networkx + networkx-metis for metis partitioning
- graphviz + python-graphviz for graph visualization

# TODOS

- string arguments for functions like nll_loss are not supported with tracing but yes with scripting
- optional inputs problem
