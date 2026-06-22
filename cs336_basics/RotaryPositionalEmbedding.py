import torch
from einops import einsum, rearrange

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,  ## Θ value for the RoPE
        d_k: int,  ## dimension of query and key vectors
        max_seq_len: int,  ## Maximum sequence length that will be inputted
        device: torch.device | None = None  ## Device to store the buffer on
    ):
        super().__init__()

        positions = torch.arange(max_seq_len, device=device)
        pair_idx = torch.arange(1, d_k // 2 + 1, device=device)
        inv_freq = 1.0 / (
                theta ** (
                2 * (pair_idx - 1) / d_k
            )
        )
        angles = torch.outer(
            positions,
            inv_freq
        )
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        ## will load automatically when torch.cuda() is called.
        ## register_buffer will by default build self.cos_cache and self.sin_cache
        ## persistent=False means the parameter will not save in checkpoint since these are not learnable parameter
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        x_arrange = rearrange(
            x,
            "... seq_len (n two) -> ... seq_len n two",
            two = 2
        )
        x1 = x_arrange[..., 0]   ## (batch, seq_len, d_k // 2)
        x2 = x_arrange[..., 1]   ## (batch, seq_len, d_k // 2)
        cos_cache = self.cos_cache[token_positions]  ## (seq_len, d_k // 2)
        sin_cache = self.sin_cache[token_positions]  ## (seq_len, d_k // 2)
        y1 = x1 * cos_cache - x2 * sin_cache
        y2 = x1 * sin_cache + x2 * cos_cache
        y_arrange = torch.stack((y1, y2), dim=-1)

        return rearrange(
            y_arrange,
            "... seq_len n two -> ... seq_len (n two)"
        )