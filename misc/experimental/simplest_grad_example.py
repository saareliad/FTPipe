import torch
m = 2
n = 3
batch = 4

a = torch.ones(batch, m)
a.requires_grad_()
layer = torch.nn.Linear(m, n, bias=False)
layer.weight.data.copy_(torch.ones(n, m))

dout_dp = torch.ones(n, m)*5   # A **given** dout_dp, detached from `out`
layer.weight.grad = dout_dp
v2 = layer(a)

# Problem: the grad is detached from the computation graph of v2.
# so the backward below not going to work
torch.autograd.backward(list(layer.parameters()), [p.grad for p in layer.parameters()])
layer.zero_grad()
print(a.grad)


# ##### Mathematically #####
m = 2
n = 3
batch = 1
a = torch.ones(batch, m)
a.requires_grad_()
layer = torch.nn.Linear(m, n, bias=False)
# layer.weight.data.copy_(torch.ones(n, m))
# v2 = layer(a)

rnd = torch.randn(v2.size())
v2 = layer(a)
(v2*rnd).sum().backward()
agc = a.grad.clone()
lgc = layer.weight.grad.clone()
a.grad.zero_()
# layer.zero_grad()

v2 = layer(a)
dp_da = layer.weight.grad.sum(0, keepdim=True).expand_as(a)
# Problem: Gradients restored from aggregated gradients, so they are all the same.
# This means that the method does not work.
print(f"restored: {dp_da}")
print(f"real: {agc}")
