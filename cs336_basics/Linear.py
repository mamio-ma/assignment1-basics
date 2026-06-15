import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from einops import einsum


class Linear(torch.nn.Module):

    in_features: int
    out_features: int
    weights: Tensor
    def __init__(
        self,
        in_features: int, ## final dimension of the input
        out_features: int, ## final dimension of the output
        device=None, ## Device to store the parameters on
        dtype=None ## Data type of the parameters
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Many machine learning papers use row-vector notation.
        #
        # This convention aligns naturally with the row-major memory layout
        # used by NumPy and PyTorch.
        #
        # Let:
        #   x ∈ R^(1 × d_in)       # input row vector
        #   W ∈ R^(d_out × d_in)   # weight matrix
        #
        # Then a linear transformation is written as:
        #
        #   y = x W^T
        #
        # Shapes:
        #   x      : (1, d_in)
        #   W      : (d_out, d_in)
        #   W.T    : (d_in, d_out)
        #   y      : (1, d_out)
        #
        # This is why PyTorch stores nn.Linear weights with shape:
        #
        #   (out_features, in_features)
        #
        # and computes:
        #
        #   output = input @ weight.T
        #
        # rather than:
        #
        #   output = weight @ input
        #
        # Example:
        #   input.shape  = (batch_size, 768)
        #   weight.shape = (3072, 768)
        #
        #   output = input @ weight.T
        #
        #   (batch_size, 768) @ (768, 3072)
        #   -> (batch_size, 3072)
        self.weights = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        std = math.sqrt(
            2 / (in_features + out_features)
        )
        ## normalize weights
        torch.nn.init.trunc_normal_(
            self.weights,
            0,
            std,
            -3 * std,
            3 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.weights,
            "... in_features, out_features in_features -> ... out_features"
        )
