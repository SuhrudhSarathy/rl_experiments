import torch

d = 0.99
a = torch.rand((128, 1))
b = torch.rand((128, 1))
c = d * a*(1-b)
print(c.shape)