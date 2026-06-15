import torch
from einops import reduce, einsum


class RMSNorm(torch.nn.Module):
    d_model: int
    eps: float
    weight: torch.Tensor  ## "gain" learnable parameter

    def __init__(
        self,
        d_model: int,  ## Hidden dimension of the model
        eps: float = 1e-5,  ## Epsilon value for numerical stability
        device=None,  ## Device to store the parameters on
        dtype=None  ## Data type of the parameters
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.empty((self.d_model, ), **factory_kwargs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = reduce(x ** 2, "batch_size sequence_length d_model -> batch_size sequence_length 1", "mean")
        rms = torch.sqrt(mean_square + self.eps)
        return x / rms * self.weight
