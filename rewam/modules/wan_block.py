from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_wan import WanAttention, WanTransformerBlock

from .dit import RMSNorm, modulate, rope_apply, scaled_dot_product_attention


class ReWAMAttentionProcessor:
    """Preserve ReWAM's RMSNorm, complex RoPE and masked SDPA calculations."""

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = attn.norm_q(attn.to_q(hidden_states))
        k = attn.norm_k(attn.to_k(context))
        v = attn.to_v(context)
        if rotary_emb is not None:
            q = rope_apply(q, rotary_emb, attn.heads)
            k = rope_apply(k, rotary_emb, attn.heads)
        output = scaled_dot_product_attention(q, k, v, attn.heads, ctx_mask=attention_mask)
        return attn.to_out[1](attn.to_out[0](output))


class ReWAMWanBlock(WanTransformerBlock):
    """Wan block with independent token/head dimensions and split MoT execution."""

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ) -> None:
        # Construct the native layout without allocating discarded attention
        # weights or consuming the caller's initialization RNG.
        device = torch.empty(0).device
        with torch.device("meta"):
            super().__init__(dim=hidden_dim, ffn_dim=ffn_dim, num_heads=num_heads, eps=eps)
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.attn1 = self._attention(hidden_dim, attn_head_dim, num_heads, eps, cross=False)
        self.attn2 = self._attention(hidden_dim, attn_head_dim, num_heads, eps, cross=True)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        # Materialize the inherited native FFN in the original Linear order.
        self.ffn.to_empty(device=device)
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)

    @staticmethod
    def _attention(dim: int, head_dim: int, heads: int, eps: float, *, cross: bool) -> WanAttention:
        attention = WanAttention(
            dim=dim,
            heads=heads,
            dim_head=head_dim,
            eps=eps,
            processor=ReWAMAttentionProcessor(),
            is_cross_attention=cross,
        )
        attention.norm_q = RMSNorm(heads * head_dim, eps=eps)
        attention.norm_k = RMSNorm(heads * head_dim, eps=eps)
        return attention

    def prepare_attention(
        self, tokens: torch.Tensor, t_mod: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        modulation = self.scale_shift_table.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        parts = modulation.chunk(6, dim=-2)
        if t_mod.ndim == 4:
            parts = tuple(part.squeeze(-2) for part in parts)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parts
        attention_input = modulate(self.norm1(tokens), shift_msa, scale_msa)
        q = self.attn1.norm_q(self.attn1.to_q(attention_input))
        k = self.attn1.norm_k(self.attn1.to_k(attention_input))
        v = self.attn1.to_v(attention_input)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        payload = {
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }
        return q, k, v, payload

    def finish_attention(
        self,
        residual: torch.Tensor,
        mixed: torch.Tensor,
        payload: dict[str, torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = residual + payload["gate_msa"] * self.attn1.to_out[0](mixed)
        tokens = tokens + self.attn2(
            self.norm2(tokens), encoder_hidden_states=context, attention_mask=context_mask
        )
        ffn_input = modulate(self.norm3(tokens), payload["shift_mlp"], payload["scale_mlp"])
        return tokens + payload["gate_mlp"] * self.ffn(ffn_input)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v, payload = self.prepare_attention(hidden_states, temb, rotary_emb)
        mixed = scaled_dot_product_attention(q, k, v, self.num_heads, ctx_mask=self_attn_mask)
        return self.finish_attention(hidden_states, mixed, payload, encoder_hidden_states, context_mask)
