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
        self.return_nested_tuple = True

    def forward(self, x):
        x = self.l0(x)
        if self.return_nested_tuple:
            return x, x + 1
        return x + 1


if __name__ == "__main__":
    model = A()
    sample = torch.randn(100, 50)
    model.l2.return_nested_tuple = True
    torch._C._jit_set_inline_everything_mode(True)
    graph = torch.jit.trace(model, sample).graph

    print("return a nested tuple")
    for node in graph.nodes():
        # we do not care about the scopes of getattr nodes
        # we can infer those ourselves
        if node.kind() == "prim::GetAttr":
            continue
        if node.scopeName() == "":
            # should trigger for the constant,multipication and TupleConstruct
            print("node without scope")
            print(node)

    model.l2.return_nested_tuple = False
    graph = torch.jit.trace(model, sample).graph

    print("\nreturn a nested tensor")
    for node in graph.nodes():
        # we do not care about the scopes of getattr nodes
        # we can infer those ourselves
        if node.kind() == "prim::GetAttr":
            continue
        if node.scopeName() == "":
            # should trigger for the constant,multipication
            print("node without scope")
            print(node)
