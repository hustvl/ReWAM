from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rewam.utils.config import as_thw


class TemporalRepresentationBottleneck(nn.Module):
    """Cross-camera attention within independent temporal groups.

    Patchify each view into lossless ``patch_size = (T,H,W)`` blocks.
    The first group contains history and current observations; later groups
    contain future frames. Groups never attend to each other, preventing
    future-to-current leakage while allowing cross-camera correspondence.

    Q/K inspect the full patch; values project directly to diffusion channels.
    Unit-normalized Gaussian value weights use the global RNG. The attention
    output projection starts at zero, preserving the linear value path.
    """

    num_cameras = 3

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
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.patch_size = as_thw(patch_size)
        self.qk_dim = int(qk_dim)
        self.num_heads = int(num_heads)
        self.norm_eps = float(norm_eps)
        self.rope_base = float(rope_base)
        self.position_scale = float(position_scale)

        if self.input_dim <= 0 or self.output_dim <= 0 or self.qk_dim <= 0:
            raise ValueError(
                "Temporal Representation Bottleneck dimensions must be positive, "
                f"got input_dim={input_dim}, output_dim={output_dim}, qk_dim={qk_dim}."
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.qk_dim % self.num_heads != 0:
            raise ValueError(f"qk_dim={self.qk_dim} must be divisible by num_heads={self.num_heads}.")
        if self.output_dim % self.num_heads != 0:
            raise ValueError(f"output_dim={self.output_dim} must be divisible by num_heads={self.num_heads}.")
        self.qk_head_dim = self.qk_dim // self.num_heads
        self.value_head_dim = self.output_dim // self.num_heads
        if self.qk_head_dim % 4 != 0:
            raise ValueError(
                "2D RoPE requires qk_dim / num_heads to be divisible by 4, "
                f"got head_dim={self.qk_head_dim}."
            )
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {norm_eps}.")
        if self.rope_base <= 0 or self.position_scale <= 0:
            raise ValueError(
                "rope_base and position_scale must be positive, " f"got {rope_base} and {position_scale}."
            )

        self.patch_dim = self.input_dim * math.prod(self.patch_size)
        self.q_projection = nn.Linear(self.patch_dim, self.qk_dim, bias=False)
        self.k_projection = nn.Linear(self.patch_dim, self.qk_dim, bias=False)
        self.value_projection = nn.Linear(self.patch_dim, self.output_dim, bias=False)
        self.output_projection = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.input_norm = nn.RMSNorm(
            self.patch_dim,
            eps=self.norm_eps,
            elementwise_affine=False,
        )
        self.camera_q_bias = nn.Parameter(torch.empty(self.num_cameras, self.num_heads, self.qk_head_dim))
        self.camera_k_bias = nn.Parameter(torch.empty(self.num_cameras, self.num_heads, self.qk_head_dim))

        with torch.no_grad():
            nn.init.normal_(self.value_projection.weight)
            self.value_projection.weight.copy_(F.normalize(self.value_projection.weight, dim=1))
            self.output_projection.weight.zero_()
            nn.init.normal_(self.camera_q_bias, std=0.02)
            nn.init.normal_(self.camera_k_bias, std=0.02)

        axis_dim = self.qk_head_dim // 2
        inv_freq = 1.0 / (
            self.rope_base ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / float(axis_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self._last_attention_entropy: torch.Tensor | None = None

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"qk_dim={self.qk_dim}, num_heads={self.num_heads}, "
            f"patch_size={self.patch_size}, "
            f"history_frames={self.patch_size[0] - 1}"
        )

    @property
    def last_attention_entropy(self) -> torch.Tensor | None:
        return self._last_attention_entropy

    def _validate_camera_features(self, camera_features: Sequence[torch.Tensor]) -> None:
        if len(camera_features) != self.num_cameras:
            raise ValueError(
                "TemporalRepresentationBottleneck requires RobotWin front/left/right features, "
                f"got {len(camera_features)} tensors."
            )
        front, left, right = camera_features
        reference = front.shape[:3]
        for camera_idx, features in enumerate(camera_features):
            if features.ndim != 5:
                raise ValueError(
                    "TemporalRepresentationBottleneck expects each camera as [B,C,T,H,W], "
                    f"camera{camera_idx}={tuple(features.shape)}."
                )
            if features.shape[:3] != reference:
                raise ValueError(
                    "TRB camera features must share [B,C,T], "
                    f"front={tuple(front.shape)} camera{camera_idx}={tuple(features.shape)}."
                )
            if int(features.shape[1]) != self.input_dim:
                raise ValueError(f"Camera {camera_idx} has C={features.shape[1]}, expected {self.input_dim}.")
            for axis, size in zip((2, 3, 4), self.patch_size):
                if int(features.shape[axis]) % size:
                    raise ValueError(
                        f"TRB requires T/H/W divisible by patch_size={self.patch_size}, "
                        f"got camera{camera_idx}={tuple(features.shape)}."
                    )
        if left.shape[-2:] != right.shape[-2:]:
            raise ValueError(
                f"RobotWin wrist feature grids must match, got {left.shape[-2:]} and {right.shape[-2:]}."
            )
        if int(front.shape[-2]) != 2 * int(left.shape[-2]):
            raise ValueError(
                "RobotWin front feature height must be twice wrist height, "
                f"got front={front.shape[-2:]} wrist={left.shape[-2:]}."
            )
        if int(front.shape[-1]) != int(left.shape[-1]) + int(right.shape[-1]):
            raise ValueError(
                "RobotWin front feature width must equal both wrist widths, "
                f"got front={front.shape[-2:]} wrists={left.shape[-2:]}/{right.shape[-2:]}."
            )

    def _patchify_camera(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        patches = (
            features.unfold(2, self.patch_size[0], self.patch_size[0])
            .unfold(3, self.patch_size[1], self.patch_size[1])
            .unfold(4, self.patch_size[2], self.patch_size[2])
        )
        batch, channels, groups, height, width, _, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
            batch,
            groups,
            height * width,
            channels * math.prod(self.patch_size),
        )
        return patches.contiguous(), (int(height), int(width))

    @staticmethod
    def _normalized_grid(
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=-1).reshape(height * width, 2)

    @staticmethod
    def _rotate_pairs(values: torch.Tensor) -> torch.Tensor:
        paired = values.unflatten(-1, (-1, 2))
        first, second = paired.unbind(dim=-1)
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def _apply_2d_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        axis_dim = self.qk_head_dim // 2
        inv_freq = self.rope_inv_freq.to(device=q.device, dtype=torch.float32)

        def apply_axis(values: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
            angles = coordinate.float().unsqueeze(-1) * self.position_scale * inv_freq
            cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype=values.dtype)
            sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype=values.dtype)
            cos = cos.view(1, 1, positions.shape[0], axis_dim)
            sin = sin.view(1, 1, positions.shape[0], axis_dim)
            return values * cos + self._rotate_pairs(values) * sin

        q_y, q_x = q.split(axis_dim, dim=-1)
        k_y, k_x = k.split(axis_dim, dim=-1)
        q = torch.cat(
            (apply_axis(q_y, positions[:, 0]), apply_axis(q_x, positions[:, 1])),
            dim=-1,
        )
        k = torch.cat(
            (apply_axis(k_y, positions[:, 0]), apply_axis(k_x, positions[:, 1])),
            dim=-1,
        )
        return q, k

    def forward(self, camera_features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        self._validate_camera_features(camera_features)
        camera_patches = []
        spatial_shapes = []
        camera_ids = []
        positions = []
        for camera_idx, features in enumerate(camera_features):
            patches, spatial_shape = self._patchify_camera(features)
            camera_patches.append(patches)
            spatial_shapes.append(spatial_shape)
            token_count = spatial_shape[0] * spatial_shape[1]
            camera_ids.append(
                torch.full((token_count,), camera_idx, device=features.device, dtype=torch.long)
            )
            positions.append(
                self._normalized_grid(
                    *spatial_shape,
                    device=features.device,
                    dtype=torch.float32,
                )
            )

        patches = torch.cat(camera_patches, dim=2)
        camera_ids_tensor = torch.cat(camera_ids, dim=0)
        positions_tensor = torch.cat(positions, dim=0)
        batch, groups, tokens, _ = patches.shape

        # Keep the normalized activation in the model dtype. Materializing an
        # explicit FP32 [B,G,N,8192] copy here would add hundreds of MB at
        # production batch sizes.
        normalized = self.input_norm(patches)

        q = (
            self.q_projection(normalized)
            .view(batch * groups, tokens, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_projection(normalized)
            .view(batch * groups, tokens, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        base = self.value_projection(patches)
        value = base.view(batch * groups, tokens, self.num_heads, self.value_head_dim).transpose(1, 2)

        q, k = self._apply_2d_rope(q, k, positions_tensor)
        camera_q_bias = self.camera_q_bias[camera_ids_tensor].permute(1, 0, 2)
        camera_k_bias = self.camera_k_bias[camera_ids_tensor].permute(1, 0, 2)
        q = q + camera_q_bias.unsqueeze(0).to(dtype=q.dtype)
        k = k + camera_k_bias.unsqueeze(0).to(dtype=k.dtype)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        mixed = torch.matmul(attention, value)
        mixed = mixed.transpose(1, 2).reshape(batch, groups, tokens, self.output_dim)
        latents = base + self.output_projection(mixed)

        entropy = -(attention.float().clamp_min(1e-12) * attention.float().clamp_min(1e-12).log())
        self._last_attention_entropy = (entropy.sum(dim=-1).mean() / math.log(max(tokens, 2))).detach()

        outputs = []
        offset = 0
        for spatial_shape in spatial_shapes:
            height, width = spatial_shape
            token_count = height * width
            camera_latents = latents[:, :, offset : offset + token_count]
            camera_latents = camera_latents.view(batch, groups, height, width, self.output_dim).permute(
                0, 4, 1, 2, 3
            )
            outputs.append(camera_latents.contiguous())
            offset += token_count
        return outputs


class CausalRunningNormalization(nn.Module):
    """Normalize TRB latents with lagged per-channel running statistics.

    The module deliberately has no affine parameters. Unlike train-mode
    BatchNorm3d, the current forward uses the running statistics *before* they
    are updated from the current batch. Consequently, the normalized clean
    observation cannot depend on future frames from the same training clip.
    Statistics are updated after normalization and synchronized across an
    initialized torch.distributed process group.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.momentum = float(momentum)
        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if not 0.0 < self.momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1], got {momentum}.")

        self.register_buffer("running_mean", torch.zeros(self.num_channels, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(self.num_channels, dtype=torch.float32))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))

    def _apply(self, fn):
        super()._apply(fn)
        # Accumulating running moments in BF16 is unnecessarily lossy. Preserve
        # their device placement while keeping floating-point buffers in FP32.
        self.running_mean = self.running_mean.float()
        self.running_var = self.running_var.float()
        return self

    def _validate_input(self, latents: torch.Tensor) -> None:
        if latents.ndim != 5:
            raise ValueError(
                "CausalRunningNormalization expects [B,C,T,H,W], " f"got {tuple(latents.shape)}."
            )
        if int(latents.shape[1]) != self.num_channels:
            raise ValueError(
                "CausalRunningNormalization channel mismatch: "
                f"got {latents.shape[1]}, expected {self.num_channels}."
            )

    @staticmethod
    def _distributed_channel_moments(
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = latents.detach().float()
        reduce_dims = (0, 2, 3, 4)
        channel_sum = values.sum(dim=reduce_dims)
        channel_sq_sum = values.square().sum(dim=reduce_dims)
        count = torch.tensor(
            values.numel() // values.shape[1],
            device=values.device,
            dtype=torch.float32,
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(channel_sum)
            torch.distributed.all_reduce(channel_sq_sum)
            torch.distributed.all_reduce(count)

        batch_mean = channel_sum / count.clamp_min(1.0)
        batch_var = channel_sq_sum / count.clamp_min(1.0) - batch_mean.square()
        return batch_mean, batch_var.clamp_min(0.0)

    @torch.no_grad()
    def _update_running_stats(self, latents: torch.Tensor) -> None:
        batch_mean, batch_var = self._distributed_channel_moments(latents)
        momentum = self.momentum
        self.running_mean.lerp_(batch_mean.to(self.running_mean), momentum)
        self.running_var.lerp_(batch_var.to(self.running_var), momentum)
        self.num_batches_tracked.add_(1)

    def forward(
        self,
        latents: torch.Tensor,
        *,
        update_stats: bool | None = None,
    ) -> torch.Tensor:
        self._validate_input(latents)
        if update_stats is None:
            update_stats = self.training

        # Clone before the in-place EMA update so autograd never observes a
        # version change in buffers used by this forward.
        running_mean = (
            self.running_mean.detach()
            .clone()
            .to(
                device=latents.device,
                dtype=torch.float32,
            )
        )
        running_var = (
            self.running_var.detach()
            .clone()
            .to(
                device=latents.device,
                dtype=torch.float32,
            )
        )
        view_shape = (1, self.num_channels, 1, 1, 1)
        normalized = (latents.float() - running_mean.view(view_shape)) / torch.sqrt(
            running_var.view(view_shape) + self.eps
        )

        if bool(update_stats):
            self._update_running_stats(latents)
        return normalized.to(dtype=latents.dtype).contiguous()

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        self._validate_input(latents)
        running_mean = self.running_mean.to(device=latents.device, dtype=torch.float32)
        running_var = self.running_var.to(device=latents.device, dtype=torch.float32)
        view_shape = (1, self.num_channels, 1, 1, 1)
        raw = latents.float() * torch.sqrt(running_var.view(view_shape) + self.eps)
        raw = raw + running_mean.view(view_shape)
        return raw.to(dtype=latents.dtype).contiguous()
