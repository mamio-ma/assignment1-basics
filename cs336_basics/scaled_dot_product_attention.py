import torch
import einops
import math
from cs336_basics.SoftMax import SoftMax

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    scores = einops.einsum(
        Q,
        K,
        "batch_size ... query_len dk, batch_size ... key_len dk -> batch_size ... query_len key_len"
    )
    scores = scores / math.sqrt(Q.shape[-1])
    ## since we want "information flows” at (i, j) pairs with value True, so here we need to inverted the attention mask
    scores = scores.masked_fill(~attention_mask, float("-inf"))
    weights = SoftMax(scores, dim=-1)
    return einops.einsum(
        weights,
        V,
        "batch_size ... query_len key_len, batch_size ... key_len dv -> batch_size ... query_len dv"
    )