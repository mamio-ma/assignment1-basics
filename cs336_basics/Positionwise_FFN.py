import torch
from numpy.f2py.auxfuncs import throw_error
from torch.nn.parameter import Parameter
from cs336_basics.Linear import Linear


class Positionwise_FFN(torch.nn.Module):
    d_model: int
    d_ff: int
    weight1: Linear
    weight2: Linear
    weight3: Linear
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if self.d_ff % 64 != 0:
            throw_error(f"{self.d_ff} should be multiple of 64")
        # self.d_ff = int(8 * d_model / 3)
        # self.d_ff = 64 * ((self.d_ff + 63) // 64)  ## make sure it is the multiple of 64
        self.weight1 = Linear(self.d_model, self.d_ff, device, dtype)
        self.weight3 = Linear(self.d_model, self.d_ff, device, dtype)
        self.weight2 = Linear(self.d_ff, self.d_model, device, dtype)

    def SiLU(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.weight1(x)
        return w1x * torch.sigmoid(w1x)

    def FFN(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight2(self.SiLU(x) * self.weight3(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.FFN(x)

