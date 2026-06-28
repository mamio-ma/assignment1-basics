import torch


def SoftMax(x: torch.Tensor, dim: int):
    x = x - x.max(dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / x.sum(dim=dim, keepdim=True)

