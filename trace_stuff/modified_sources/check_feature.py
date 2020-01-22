import torch
import torch.nn as nn


class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.l1 = nn.Linear(50, 50)
        self.l2 = B()

    def forward(self, x):
        x = self.l1(x)

        # in this line both the constant 2 and tmp will not have a scopeName()
        tmp = x * 2

        # this line will vary depends if we return a tuple or not
        return self.l2(tmp)


class B(nn.Module):
    def __init__(self):
        super(B, self).__init__()
        self.l0 = nn.Linear(50, 50)

    def forward(self, x):
        x = self.l0(x)
        return x + 1


if __name__ == "__main__":
    model = A()
    sample = torch.randn(100, 50)
    graph = torch.jit.trace(model, sample).graph

    print(
        "minimal graph expect to see prim::CallMethod[forward](l1) prim::CallMethod[forward](l2)")
    print(graph)
    with open("traces.txt", "w") as f:
        f.write("minimal trace should have callMethod for A.l1,A.l2\n")
        f.write(str(graph))
    print(
        "\nverbose graph is expect to see no prim::CallMethod[forward] at all")
    # modifies inplace
    torch._C._jit_pass_inline(graph, 10)
    print(graph)
    with open("traces.txt", "a") as f:
        f.write("\nmaximal trace should have no CallMethod at all\n")
        f.write(str(graph))

    graph = torch.jit.trace(model, sample).graph
    print(
        "\ndepth 1 graph expect to see something like CallMethod[forward](A.l2.l0)")
    torch._C._jit_pass_inline(graph, 1)
    print(graph)
    with open("traces.txt", "a") as f:
        f.write("\ndepth 1 trace should have CallMethod A.l2.l0\n")
        f.write(str(graph))
