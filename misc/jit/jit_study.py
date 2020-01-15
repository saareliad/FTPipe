import torch

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        # This parameter will be copied to the new ScriptModule
        self.weight = torch.nn.Parameter(torch.rand(N, M))

        # When this submodule is used, it will be compiled
        self.linear = torch.nn.Linear(N, M)

    def forward(self, input):
        output = self.weight.mv(input)

        # This calls the `forward` method of the `nn.Linear` module, which will
        # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
        output = self.linear(output)
        return output, output+3

# Create traces with 3 different methods.
scripted_module = torch.jit.script(MyModule(2, 3))
traced = torch.jit.trace(MyModule(2,3),torch.randn(3))

if hasattr(torch.jit, "get_trace_graph"):
    get_trace_graph = torch.jit.get_trace_graph
else:
    assert hasattr(torch.jit, "_get_trace_graph")
    get_trace_graph = torch.jit._get_trace_graph

trace_graph, _ = get_trace_graph(MyModule(2,3), torch.randn(3))

# Print everything
print("torch.jit.trace:")
print(traced.graph)

print()

print("torch.jit.script")
print(scripted_module.graph)

print()

print("torch.jit._get_trace_graph")
print(trace_graph)

