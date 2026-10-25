from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from .dit import (
    Head,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)
from .mot import MoTExpertState, MoTLayerIO, MoTTokenLayout
from .wan_block import ReWAMWanBlock


class WorldDiT(nn.Module):
    """Wan2.2 video DiT expert implementing the MoT expert protocol.

    A Conv3d patch embedding feeds Wan blocks and a modulated output head.
    The current latent frame stays clean (t=0); future frames carry sampled
    timesteps independently.
    """

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.eps = eps
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers

        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
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
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        device = self.patch_embedding.weight.device
        if any(freqs.is_meta for freqs in self.freqs) and device.type != "meta":
            with torch.device("cpu"):
                self.freqs = precompute_freqs_cis_3d(self.attn_head_dim)
        self.freqs = tuple(freqs.to(device=device) for freqs in self.freqs)
        return result

    @property
    def mot_num_layers(self) -> int:
        return self.num_layers

    @property
    def mot_num_heads(self) -> int:
        return self.num_heads

    @property
    def mot_head_dim(self) -> int:
        return self.attn_head_dim

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)

    def unpatchify(self, x: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-frame video self-attention mask (matches reference)."""
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if video_seq_len % video_tokens_per_frame:
            raise ValueError("World sequence length must be divisible by tokens per frame.")
        num_frames = video_seq_len // video_tokens_per_frame
        causal = torch.tril(torch.ones((num_frames, num_frames), dtype=torch.bool, device=device))
        return causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
            video_tokens_per_frame, dim=1
        )

    def _embed_latents(
        self, video_latents: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int], int]:
        if video_latents.ndim != 5:
            raise ValueError(f"`video_latents` must be 5D [B, C, T, H, W], got {tuple(video_latents.shape)}")
        batch_size = video_latents.shape[0]
        if timestep.ndim not in (1, 2):
            raise ValueError(f"`timestep` must be [B] or [B,T], got {tuple(timestep.shape)}.")
        if timestep.shape[0] != batch_size:
            raise ValueError(f"Timestep batch {timestep.shape[0]} does not match video batch {batch_size}.")
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if video_latents.shape[3] % patch_h != 0 or video_latents.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({video_latents.shape[3]}, {video_latents.shape[4]}), "
                f"patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (video_latents.shape[3] // patch_h) * (video_latents.shape[4] // patch_w)

        x = self.patchify(video_latents)
        f, h, w = x.shape[2:]
        # Per-token (separated) timestep: first latent frame t=0, others = timestep.
        if timestep.ndim == 1:
            frame_timesteps = timestep.view(batch_size, 1).expand(-1, x.shape[2])
        else:
            if timestep.shape[1] != x.shape[2]:
                raise ValueError(
                    f"Per-frame timestep T={timestep.shape[1]} does not match latent T={x.shape[2]}."
                )
            frame_timesteps = timestep
        token_timesteps = frame_timesteps.unsqueeze(-1).expand(-1, -1, tokens_per_frame).clone()
        token_timesteps[:, 0, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, -1)
        token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)  # [B,S,D]
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))  # [B,S,6,D]

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x_tokens, t, t_mod, (f, h, w), tokens_per_frame

    def prepare_step(
        self, video_latents: torch.Tensor, timestep: torch.Tensor, conditioning: MoTExpertState
    ) -> MoTExpertState:
        """Update world tokens/time modulation with fixed request-local conditioning."""
        tokens, t, t_mod, grid_size, tokens_per_frame = self._embed_latents(video_latents, timestep)
        return MoTExpertState(
            tokens=tokens,
            layout=conditioning.layout,
            payload={
                **conditioning.payload,
                "t_mod": t_mod,
                "head_t_mod": t,
                "grid_size": grid_size,
                "tokens_per_frame": tokens_per_frame,
            },
        )

    def prepare(
        self,
        video_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        valid_mask: torch.Tensor,
        frame_ids: torch.Tensor,
    ) -> MoTExpertState:
        """Tokenize video latents into a MoT expert state.

        Args:
            video_latents: ``[B, in_dim, T, H, W]`` compressed representation states.
            timestep: ``[B]`` (per-sample) or ``[B,T]`` (per-frame). First
                latent frame is forced to ``t=0`` regardless.
            context: text/proprio context ``[B, L, text_dim]``.
            context_mask: ``[B, L]`` bool.
            valid_mask: ``[B, S_v]`` bool over video tokens (excludes padding).
            frame_ids: ``[B, S_v]`` int64 latent-frame id per token (``0..T_lat-1``).
        """
        x_tokens, t, t_mod, (f, h, w), tokens_per_frame = self._embed_latents(video_latents, timestep)
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

        if context_mask is None:
            context_mask = torch.ones(
                (context.shape[0], context.shape[1]),
                dtype=torch.bool,
                device=context.device,
            )
        # Cross-attention context mask expanded over the video token sequence.
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, x_tokens.shape[1], -1)

        context_emb = self.text_embedding(context)
        layout = MoTTokenLayout(
            groups={"video": valid_mask.to(x_tokens.device)},
            frame_ids=frame_ids.to(x_tokens.device),
            valid_mask=valid_mask.to(x_tokens.device),
        )
        payload = {
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "freqs": freqs,
            "grid_size": (f, h, w),
            "head_t_mod": t,  # [B,S,D] for Head.forward 3D path
            "tokens_per_frame": tokens_per_frame,
        }
        return MoTExpertState(tokens=x_tokens, layout=layout, payload=payload)

    def mot_layer_qkv(self, layer_idx: int, state: MoTExpertState) -> MoTLayerIO:
        block = self.blocks[layer_idx]
        x = state.tokens
        t_mod = state.payload["t_mod"]
        freqs = state.payload["freqs"]

        q, k, v, layer_payload = block.prepare_attention(x, t_mod, freqs)
        # Reshape to [B,H,S,Dh] at the protocol boundary (pure view after RoPE).
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
        # attention_output: [B,H,S,Dh] -> [B,S,H*Dh]
        B, H, S, Dh = attention_output.shape
        mixed_flat = attention_output.permute(0, 2, 1, 3).reshape(B, S, H * Dh)

        p = layer_io.payload
        x = block.finish_attention(layer_io.residual, mixed_flat, p, p["context"], p["context_mask"])
        return MoTExpertState(tokens=x, layout=state.layout, payload=state.payload)

    def mot_finalize(self, state: MoTExpertState) -> torch.Tensor:
        grid_size = state.payload["grid_size"]
        x = self.head(state.tokens, state.payload["head_t_mod"])
        x = self.unpatchify(x, grid_size)
        return x

    # MoT uses the layer protocol; direct calls run this expert independently.
    def forward(
        self,
        video_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        frame_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            B, _, T, H, W = video_latents.shape
            patch_h, patch_w = int(self.patch_size[1]), int(self.patch_size[2])
            tokens_per_frame = (H // patch_h) * (W // patch_w)
            S = (T) * tokens_per_frame
            valid_mask = torch.ones((B, S), dtype=torch.bool, device=video_latents.device)
        if frame_ids is None:
            B, _, T, H, W = video_latents.shape
            patch_h, patch_w = int(self.patch_size[1]), int(self.patch_size[2])
            tokens_per_frame = (H // patch_h) * (W // patch_w)
            # latent-frame id per token: repeat each frame id across its spatial tokens.
            frame_ids = (
                torch.arange(T, device=video_latents.device)
                .view(1, T, 1)
                .expand(B, T, tokens_per_frame)
                .reshape(B, -1)
            )
        state = self.prepare(video_latents, timestep, context, context_mask, valid_mask, frame_ids)
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=state.tokens.shape[1],
            video_tokens_per_frame=int(state.payload["tokens_per_frame"]),
            device=state.tokens.device,
        )
        for layer_idx in range(self.num_layers):
            io = self.mot_layer_qkv(layer_idx, state)
            from .dit import scaled_dot_product_attention

            mixed = scaled_dot_product_attention(
                io.q, io.k, io.v, num_heads=self.num_heads, ctx_mask=self_attn_mask
            )
            state = self.mot_layer_post_attention(layer_idx, mixed, io, state)
        return self.mot_finalize(state)
