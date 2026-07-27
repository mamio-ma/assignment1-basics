import einops
import torch

from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadAttention(torch.nn.Module):
    d_model: int  ## Dimensionality of the Transformer block inputs.
    num_heads: int  ##  ## Number of heads to use in multi-head self-attention.
    d: int  ## dimention of the weight (Q, K, V)
    rope: RotaryPositionalEmbedding | None
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d = self.d_model // self.num_heads
        if theta:
            self.rope = RotaryPositionalEmbedding(theta, self.d, 4096)

    def _rearrange(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return einops.rearrange(
            x,
            "batch seq_len (heads d) -> batch heads seq_len d",
            heads=self.num_heads
        )

    def forward(
            self,
            Wq: torch.Tensor,
            Wk: torch.Tensor,
            Wv: torch.Tensor,
            Wo: torch.Tensor,
            x: torch.Tensor,
            token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        Wq = self._rearrange(x @ Wq.T)
        Wk = self._rearrange(x @ Wk.T)
        if token_positions is not None:
            Wq = self.rope(Wq, token_positions)
            Wk = self.rope(Wk, token_positions)
        Wv = self._rearrange(x @ Wv.T)
        # example:
        # tensor([
        #     [True, False, False, False],
        #     [True, True, False, False],
        #     [True, True, True, False],
        #     [True, True, True, True]
        # ])
        causal_mask = torch.tril(
            torch.ones(x.shape[-2], x.shape[-2], dtype=torch.bool)
        )
        return einops.rearrange(
            scaled_dot_product_attention(Wq, Wk, Wv, attention_mask=causal_mask),
            "batch heads seq_len d -> batch seq_len (heads d)"
        ) @ Wo.T