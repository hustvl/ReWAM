from __future__ import annotations

import torch
import torch.nn as nn

from .dit import (
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)
from .mot import MoTExpertState, MoTLayerIO, MoTTokenLayout
from .wan_block import ReWAMWanBlock


class ActionDiT(nn.Module):
    """Action diffusion transformer expert implementing the MoT expert protocol."""

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.eps = eps

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [ReWAMWanBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        device = self.action_encoder.weight.device
        if self.freqs.is_meta and device.type != "meta":
            with torch.device("cpu"):
                self.freqs = precompute_freqs_cis(self.attn_head_dim, end=1024)
        # Do not cast complex RoPE tables with the model's real-valued weights.
        self.freqs = self.freqs.to(device=device)
        return result

    def _embed_tokens(
        self, action_tokens: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        return self.action_encoder(action_tokens), t_mod

    def prepare_step(
        self, action_tokens: torch.Tensor, timestep: torch.Tensor, conditioning: MoTExpertState
    ) -> MoTExpertState:
        """Update denoising tokens while reusing request-local conditioning/layout."""
        tokens, t_mod = self._embed_tokens(action_tokens, timestep)
        return MoTExpertState(
            tokens=tokens, layout=conditioning.layout, payload={**conditioning.payload, "t_mod": t_mod}
        )

    @property
    def mot_num_layers(self) -> int:
        return self.num_layers

    @property
    def mot_num_heads(self) -> int:
        return self.num_heads

    @property
    def mot_head_dim(self) -> int:
        return self.attn_head_dim

    def prepare(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        valid_mask: torch.Tensor | None = None,
    ) -> MoTExpertState:
        """Tokenize raw action tokens into a MoT expert state.

        Args:
            action_tokens: ``[B, S, action_dim]`` (noised action latents).
            timestep: ``[B]`` per-sample timestep.
            context: cross-attention context ``[B, L, text_dim]`` (text/proprio).
            context_mask: ``[B, L]`` bool.
            valid_mask: ``[B, S]`` bool over action tokens (excludes padding).
        """
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, S, action_dim], got {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got {tuple(context.shape)}")
        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(f"Batch mismatch action vs context: {batch_size} vs {context.shape[0]}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2 or context_mask.shape != (
                batch_size,
                context.shape[1],
            ):
                got = tuple(context_mask.shape)
                raise ValueError(
                    f"`context_mask` must be [B, L] matching context, got {got} "
                    f"vs {(batch_size, context.shape[1])}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}.")

        tokens, t_mod = self._embed_tokens(action_tokens, timestep)
        context_emb = self.text_embedding(context)  # [B, L, D]
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [B, S, L]
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1)  # [S, 1, rope_dim]

        if valid_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=tokens.device)
        layout = MoTTokenLayout(
            groups={"action": valid_mask.to(tokens.device)},
            frame_ids=None,
            valid_mask=valid_mask.to(tokens.device),
        )
        payload = {
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "freqs": freqs,
        }
        return MoTExpertState(tokens=tokens, layout=layout, payload=payload)

    def mot_layer_qkv(self, layer_idx: int, state: MoTExpertState) -> MoTLayerIO:
        block = self.blocks[layer_idx]
        x = state.tokens
        t_mod = state.payload["t_mod"]
        freqs = state.payload["freqs"]

        q, k, v, layer_payload = block.prepare_attention(x, t_mod, freqs)
        B, S = q.shape[0], q.shape[1]
        q = q.view(B, S, self.num_heads, self.attn_head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, S, self.num_heads, self.attn_head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, S, self.num_heads, self.attn_head_dim).permute(0, 2, 1, 3).contiguous()

        layer_payload.update(context=state.payload["context"], context_mask=state.payload["context_mask"])
        return MoTLayerIO(q=q, k=k, v=v, residual=x, payload=layer_payload)

    def mot_layer_post_attention(
        self,
        layer_idx: int,
        attention_output: torch.Tensor,
        layer_io: MoTLayerIO,
        state: MoTExpertState,
    ) -> MoTExpertState:
        block = self.blocks[layer_idx]
        B, H, S, Dh = attention_output.shape
        mixed_flat = attention_output.permute(0, 2, 1, 3).reshape(B, S, H * Dh)

        p = layer_io.payload
        x = block.finish_attention(layer_io.residual, mixed_flat, p, p["context"], p["context_mask"])
        return MoTExpertState(tokens=x, layout=state.layout, payload=state.payload)

    def mot_finalize(self, state: MoTExpertState) -> torch.Tensor:
        return self.head(state.tokens)  # [B, S, action_dim]

    # MoT uses the layer protocol; direct calls run this expert independently.
    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self.prepare(action_tokens, timestep, context, context_mask, valid_mask)
        for layer_idx in range(self.num_layers):
            io = self.mot_layer_qkv(layer_idx, state)
            from .dit import scaled_dot_product_attention

            mixed = scaled_dot_product_attention(io.q, io.k, io.v, num_heads=self.num_heads, ctx_mask=None)
            state = self.mot_layer_post_attention(layer_idx, mixed, io, state)
        return self.mot_finalize(state)
