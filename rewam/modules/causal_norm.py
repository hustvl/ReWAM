from __future__ import annotations

import torch
import torch.nn as nn


class CausalRunningNormalization(nn.Module):
    """Normalize latents with lagged per-channel running statistics.

    The current forward uses the running moments *before* updating them from
    the current batch.  This keeps the normalized current observation
    independent of future frames that happen to be packed into the same
    training clip.  Moments are accumulated in FP32 and synchronized across
    an initialized distributed process group.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
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
        # Keep running moments in FP32 even when the model is converted to BF16.
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
        self.running_mean.lerp_(batch_mean.to(self.running_mean), self.momentum)
        self.running_var.lerp_(batch_var.to(self.running_var), self.momentum)
        self.num_batches_tracked.add_(1)

    def forward(self, latents: torch.Tensor, *, update_stats: bool | None = None) -> torch.Tensor:
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


__all__ = ["CausalRunningNormalization"]
