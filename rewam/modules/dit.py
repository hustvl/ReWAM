from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention on top of ``F.scaled_dot_product_attention``.

    Args:
        q, k, v: Either ``[B, S, H*Dh]`` (expert-internal, reference layout) or
            ``[B, H, S, Dh]`` (MoT coupling-layer layout).  When 4D the SDPA is
            called directly without any rearrange.
        num_heads: Number of attention heads ``H`` (only needed for the 3D path).
        ctx_mask: Optional bool attention mask where ``True`` = keep.  For the 3D
            path it is broadcast to ``[B, 1, S_q, S_k]``; for the 4D path it must
            already be broadcastable to ``[B, H, S_q, S_k]`` (typically
            ``[B, 1, S_q, S_k]``).

    Returns:
        Output in the *same* layout convention as the input ``q``.
    """
    if q.dim() == 4:
        # MoT coupling-layer path: already [B, H, S, Dh].
        return F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    # Standalone-expert path: [B, S, H*Dh] -> [B, H, S, Dh] -> back.
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    # Broadcast a per-sample cross-attention mask over all heads.
    if ctx_mask is not None and ctx_mask.dim() == 3:
        ctx_mask = ctx_mask.unsqueeze(1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
    out = rearrange(out, "b n s d -> b s (n d)", n=num_heads)
    return out


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    # Retain complex64 RoPE independently of the model parameter dtype.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = torch.outer(torch.arange(end, dtype=torch.float32, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    # Rotate in complex64, then restore the input dtype.
    x_out = torch.view_as_complex(x.float().reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(
                2, dim=1
            )
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x
