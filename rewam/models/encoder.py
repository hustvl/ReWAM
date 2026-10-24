from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn


class DINOEncoder(nn.Module):
    """Encode [B,3,T,H,W] videos into one [B,C,T,h,w] grid per requested layer.

    The frozen backbone is loaded lazily and kept outside module registration,
    so ReWAM checkpoints and parameter groups contain only the learned model.
    ``output_dim`` specifies the channel count needed to construct TRB before
    loading the backbone; the loaded model must have the same channel count.
    """

    def __init__(
        self,
        *,
        model_name: str = "dinov3_vitl16",
        hub_repo: str = "facebookresearch/dinov3:main",
        output_dim: int = 1024,
        hub_source: str = "github",
        hub_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.hub_repo = hub_repo
        self.output_dim = int(output_dim)
        self.hub_source = hub_source
        self.hub_kwargs = dict(hub_kwargs or {})
        self.patch_size: int | None = None
        self.register_buffer("_dtype_anchor", torch.empty(0), persistent=False)
        object.__setattr__(self, "_backbone", None)
        self.eval()

    @property
    def backbone(self) -> nn.Module | None:
        return self.__dict__["_backbone"]

    def train(self, mode: bool = True):
        super().train(False)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        if self.backbone is not None:
            self.backbone._apply(fn)
        return self

    def _load(self, device: torch.device) -> nn.Module:
        if self.backbone is None:
            backbone = torch.hub.load(
                self.hub_repo,
                self.model_name,
                source=self.hub_source,
                **self.hub_kwargs,
            )
            if not hasattr(backbone, "get_intermediate_layers"):
                raise ValueError("DINO encoder requires get_intermediate_layers().")
            if int(backbone.embed_dim) != self.output_dim:
                raise ValueError(
                    f"DINO embed_dim={backbone.embed_dim} does not match vision_encoder.output_dim={self.output_dim}."
                )
            patch_size = getattr(backbone, "patch_size", None)
            if patch_size is None:
                patch_size = backbone.patch_embed.patch_size
            if isinstance(patch_size, (tuple, list)):
                if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                    raise ValueError(f"DINO requires square patches, got {patch_size}.")
                patch_size = patch_size[0]
            self.patch_size = int(patch_size)
            backbone.eval().requires_grad_(False)
            backbone.to(device=device, dtype=self._dtype_anchor.dtype)
            object.__setattr__(self, "_backbone", backbone)
        else:
            parameter = next(self.backbone.parameters(), None)
            if parameter is not None and (
                parameter.device != device or parameter.dtype != self._dtype_anchor.dtype
            ):
                self.backbone.to(device=device, dtype=self._dtype_anchor.dtype)
        return self.backbone

    @torch.no_grad()
    def forward(self, video: torch.Tensor, *, layers: Sequence[int]) -> tuple[torch.Tensor, ...]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"DINO video must be [B,3,T,H,W], got {tuple(video.shape)}.")
        backbone = self._load(video.device)
        batch, _, frames, height, width = video.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"DINO input HxW must be divisible by patch_size={self.patch_size}.")
        images = (video.detach().float() + 1.0) * 0.5
        images = images.clamp(0.0, 1.0).permute(0, 2, 1, 3, 4).reshape(batch * frames, 3, height, width)
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        images = ((images - mean) / std).to(dtype=self._dtype_anchor.dtype)
        outputs = backbone.get_intermediate_layers(
            images,
            n=list(layers),
            reshape=False,
            return_class_token=False,
            norm=False,
        )
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(layers):
            raise ValueError("DINO must return one feature tensor per requested layer.")
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        features = []
        for tokens in outputs:
            if isinstance(tokens, (tuple, list)):
                tokens = tokens[0]
            if tokens.ndim != 3 or tokens.shape[-1] != self.output_dim:
                raise ValueError(
                    f"DINO patch tokens must be [B*T,N,{self.output_dim}], got {tuple(tokens.shape)}."
                )
            tokens = tokens[:, -(grid_h * grid_w) :]
            features.append(
                tokens.reshape(batch, frames, grid_h, grid_w, self.output_dim).permute(0, 4, 1, 2, 3)
            )
        return tuple(features)
