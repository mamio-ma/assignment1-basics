import torch
from torch import Tensor
from einops import einsum
from torch.nn.parameter import Parameter


class Embedding(torch.nn.Module):

    num_embeddings: int
    embedding_dim: int
    weight: Tensor
    def __init__(
        self,
        num_embeddings,  ## Size of the vocabulary
        embedding_dim,  ## Dimension of the embedding vectors, i.e., dmodel
        device=None,  ## Device to store the parameters on
        dtype=None  ## Data type of the parameters
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(
            self.weight,
            0,
            1,
            -3,
            3
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]