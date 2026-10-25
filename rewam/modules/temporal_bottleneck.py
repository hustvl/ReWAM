from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import as_thw


class TemporalRepresentationBottleneck(nn.Module):
    """Compress temporal groups using independent cross-view attention.

    Each view is losslessly patchified into ``patch_size = (T,H,W)``
    blocks. Tokens attend jointly across cameras within a group, never across
    groups. Results remain separate per view for the RoboTwin spatial layout.
    Unit-normalized Gaussian value weights use the global RNG; a zero-initialized
    output projection makes the initial result equal to the linear value path.
    """

    num_views = 3

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        patch_size: Sequence[int] = (2, 2, 2),
        qk_dim: int = 1024,
        num_heads: int = 8,
        norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        position_scale: float = 16.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.patch_size = as_thw(patch_size)
        self.qk_dim = int(qk_dim)
        self.num_heads = int(num_heads)
        self.norm_eps = float(norm_eps)
        self.rope_base = float(rope_base)
        self.position_scale = float(position_scale)
        if min(self.input_dim, self.output_dim, self.qk_dim, self.num_heads) <= 0:
            raise ValueError("Tokenizer dimensions and num_heads must be positive.")
        if self.qk_dim % self.num_heads or self.output_dim % self.num_heads:
            raise ValueError("qk_dim and output_dim must be divisible by num_heads.")
        self.query_key_head_dim = self.qk_dim // self.num_heads
        self.value_head_dim = self.output_dim // self.num_heads
        if self.query_key_head_dim % 4:
            raise ValueError("2D RoPE requires qk_dim/num_heads divisible by 4.")

        self.patch_dim = self.input_dim * math.prod(self.patch_size)
        self.input_norm = nn.RMSNorm(self.patch_dim, eps=self.norm_eps, elementwise_affine=False)
        self.query = nn.Linear(self.patch_dim, self.qk_dim, bias=False)
        self.key = nn.Linear(self.patch_dim, self.qk_dim, bias=False)
        self.value = nn.Linear(self.patch_dim, self.output_dim, bias=False)
        self.output = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.view_query_bias = nn.Parameter(
            torch.empty(self.num_views, self.num_heads, self.query_key_head_dim)
        )
        self.view_key_bias = nn.Parameter(
            torch.empty(self.num_views, self.num_heads, self.query_key_head_dim)
        )

        with torch.no_grad():
            nn.init.normal_(self.value.weight)
            self.value.weight.copy_(F.normalize(self.value.weight, dim=1))
            self.output.weight.zero_()
            nn.init.normal_(self.view_query_bias, std=0.02)
            nn.init.normal_(self.view_key_bias, std=0.02)

        axis_dim = self.query_key_head_dim // 2
        inv_freq = 1.0 / (
            self.rope_base ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / float(axis_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self._last_attention_entropy: torch.Tensor | None = None

    @property
    def last_attention_entropy(self) -> torch.Tensor | None:
        return self._last_attention_entropy

    def _validate_views(self, features: Sequence[torch.Tensor]) -> None:
        if len(features) != self.num_views:
            raise ValueError(f"Tokenizer requires front/left/right features, got {len(features)} views.")
        front, left, right = features
        for index, feature in enumerate(features):
            if feature.ndim != 5:
                raise ValueError(f"View {index} must be [B,C,T,H,W], got {tuple(feature.shape)}.")
            if feature.shape[:3] != front.shape[:3] or feature.shape[1] != self.input_dim:
                raise ValueError("All tokenizer views must share [B,C,T] and configured channels.")
            if any(feature.shape[axis] % size for axis, size in zip((2, 3, 4), self.patch_size)):
                raise ValueError(
                    f"TRB requires T/H/W divisible by patch_size={self.patch_size}, "
                    f"got view {index}: {tuple(feature.shape)}."
                )
        if left.shape[-2:] != right.shape[-2:]:
            raise ValueError("Left and right wrist feature grids must match.")
        if front.shape[-2] != 2 * left.shape[-2] or front.shape[-1] != 2 * left.shape[-1]:
            raise ValueError(
                "Front feature grid must be twice each wrist grid in both dimensions, "
                f"got front={front.shape[-2:]}, wrist={left.shape[-2:]}."
            )

    def _patchify(self, feature: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        patches = (
            feature.unfold(2, self.patch_size[0], self.patch_size[0])
            .unfold(3, self.patch_size[1], self.patch_size[1])
            .unfold(4, self.patch_size[2], self.patch_size[2])
        )
        batch, channels, groups, height, width, _, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
            batch, groups, height * width, channels * math.prod(self.patch_size)
        )
        return patches.contiguous(), (int(height), int(width))

    @staticmethod
    def _grid(height: int, width: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, height, device=device)
        x = torch.linspace(0.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=-1).reshape(height * width, 2)

    @staticmethod
    def _rotate_pairs(values: torch.Tensor) -> torch.Tensor:
        first, second = values.unflatten(-1, (-1, 2)).unbind(-1)
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def _rope(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        axis_dim = self.query_key_head_dim // 2
        inv_freq = self.rope_inv_freq.to(device=values.device)

        def apply_axis(axis_values: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
            angle = coordinate.float().unsqueeze(-1) * self.position_scale * inv_freq
            cos = torch.repeat_interleave(angle.cos(), 2, dim=-1).to(axis_values.dtype)
            sin = torch.repeat_interleave(angle.sin(), 2, dim=-1).to(axis_values.dtype)
            cos = cos.view(1, 1, positions.shape[0], axis_dim)
            sin = sin.view(1, 1, positions.shape[0], axis_dim)
            return axis_values * cos + self._rotate_pairs(axis_values) * sin

        y_values, x_values = values.split(axis_dim, dim=-1)
        return torch.cat(
            (
                apply_axis(y_values, positions[:, 0]),
                apply_axis(x_values, positions[:, 1]),
            ),
            dim=-1,
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        self._validate_views(features)
        view_patches = []
        spatial_shapes = []
        view_ids = []
        positions = []
        for view_index, feature in enumerate(features):
            patches, shape = self._patchify(feature)
            view_patches.append(patches)
            spatial_shapes.append(shape)
            count = shape[0] * shape[1]
            view_ids.append(torch.full((count,), view_index, device=feature.device, dtype=torch.long))
            positions.append(self._grid(*shape, device=feature.device))

        patches = torch.cat(view_patches, dim=2)
        view_ids_tensor = torch.cat(view_ids)
        positions_tensor = torch.cat(positions)
        batch, groups, tokens, _ = patches.shape
        normalized = self.input_norm(patches)
        query = (
            self.query(normalized)
            .view(batch * groups, tokens, self.num_heads, self.query_key_head_dim)
            .transpose(1, 2)
        )
        key = (
            self.key(normalized)
            .view(batch * groups, tokens, self.num_heads, self.query_key_head_dim)
            .transpose(1, 2)
        )
        base = self.value(patches)
        value = base.view(batch * groups, tokens, self.num_heads, self.value_head_dim).transpose(1, 2)

        query = self._rope(query, positions_tensor)
        key = self._rope(key, positions_tensor)
        query = query + self.view_query_bias[view_ids_tensor].permute(1, 0, 2).unsqueeze(0).to(query.dtype)
        key = key + self.view_key_bias[view_ids_tensor].permute(1, 0, 2).unsqueeze(0).to(key.dtype)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.query_key_head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        mixed = torch.matmul(attention, value).transpose(1, 2).reshape(batch, groups, tokens, self.output_dim)
        states = base + self.output(mixed)
        entropy = -(attention.float().clamp_min(1e-12) * attention.float().clamp_min(1e-12).log())
        self._last_attention_entropy = (entropy.sum(-1).mean() / math.log(max(tokens, 2))).detach()

        result = []
        offset = 0
        for height, width in spatial_shapes:
            count = height * width
            state = states[:, :, offset : offset + count]
            result.append(
                state.reshape(batch, groups, height, width, self.output_dim)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )
            offset += count
        return result
