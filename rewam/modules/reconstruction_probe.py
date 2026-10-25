from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import as_thw

logger = logging.getLogger(__name__)


class ReconstructionProbe(nn.Module):
    """Restore detached TRB states to DINO features, with optional RGB decoding.

    Only the TRB-shaped transposed projection is registered and trained. The frozen
    external RAEv2 decoder is loaded lazily, kept outside checkpoint/optimizer
    traversal, and managed here for device, dtype, eval mode and lifetime.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        patch_size: tuple[int, int, int] = (2, 2, 2),
        decoder_path: str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if min(self.input_dim, self.output_dim) <= 0:
            raise ValueError("Reconstruction probe dimensions must be positive.")
        patch_size = as_thw(patch_size)
        self.projection = nn.ConvTranspose3d(
            self.input_dim, self.output_dim, kernel_size=patch_size, stride=patch_size
        )
        self.decoder_path = decoder_path or None
        object.__setattr__(self, "_decoder", None)

    @property
    def decoder(self) -> nn.Module | None:
        return self.__dict__["_decoder"]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.decoder is not None:
            self.decoder.eval()
        return self

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        if self.decoder is not None:
            self.decoder._apply(fn)
        return self

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 5 or states.shape[1] != self.input_dim:
            raise ValueError(
                f"Reconstruction probe expects [B,{self.input_dim},T,H,W], got {tuple(states.shape)}."
            )
        return self.projection(states.detach()).contiguous()

    @torch.no_grad()
    def decode_rgb(self, features: torch.Tensor) -> torch.Tensor:
        """Decode one camera's [B,C,T,H,W] feature grid to [B*T,3,256,256] RGB."""
        if features.ndim != 5 or features.shape[1] != self.output_dim:
            raise ValueError(
                f"RGB decoding expects [B,{self.output_dim},T,H,W], got {tuple(features.shape)}."
            )
        if self.decoder is None:
            if self.decoder_path is None:
                raise ValueError("Set reconstruction_probe.decoder_path to the RAEv2 decoder checkpoint.")
            decoder = RAEv2GeneralDecoder.from_checkpoint(
                self.decoder_path, device=features.device, dtype=self.projection.weight.dtype
            )
            decoder.requires_grad_(False).eval()
            object.__setattr__(self, "_decoder", decoder)
            logger.info("Loaded frozen RAEv2 decoder: path=%s", self.decoder_path)

        batch, channels, frames, height, width = features.shape
        images = features.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        images = F.interpolate(images.float(), size=(16, 16), mode="bilinear", align_corners=False)
        decoder_parameter = next(self.decoder.parameters(), None)
        # Accelerate may return FP32 features; follow the decoder parameter dtype.
        if decoder_parameter is not None:
            images = images.to(device=decoder_parameter.device, dtype=decoder_parameter.dtype)
        return self.decoder(images).float().clamp(0.0, 1.0)

    def release_decoder(self) -> None:
        """Release the external RGB decoder without touching the learned probe."""
        object.__setattr__(self, "_decoder", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}.")
        self.num_attention_heads = int(num_heads)
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(hidden_size, hidden_size, bias=True)
        self.value = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(0.0)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        return value.view(shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        query = self._split_heads(self.query(hidden_states))
        key = self._split_heads(self.key(hidden_states))
        value = self._split_heads(self.value(hidden_states))
        attention = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention = self.dropout(attention.softmax(dim=-1))
        context = torch.matmul(attention, value).permute(0, 2, 1, 3).contiguous()
        return (context.view(*context.shape[:-2], self.all_head_size),)


class _SelfOutput(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states))


class _Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = _SelfAttention(hidden_size, num_heads)
        self.output = _SelfOutput(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(hidden_states)[0])


class _Intermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden_states))


class _Output(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.0)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states)) + residual


class _DecoderLayer(nn.Module):
    """Parameter-compatible copy of the ViT-MAE block used by RAEv2."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, eps: float):
        super().__init__()
        self.attention = _Attention(hidden_size, num_heads)
        self.intermediate = _Intermediate(hidden_size, intermediate_size)
        self.output = _Output(hidden_size, intermediate_size)
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.layernorm_before(hidden_states))
        return self.output(self.intermediate(self.layernorm_after(hidden_states)), hidden_states)


class RAEv2GeneralDecoder(nn.Module):
    """Released RAEv2 DINOv3-L K7 general decoder architecture."""

    latent_dim = 1024
    decoder_hidden_size = 1152
    decoder_intermediate_size = 4096
    decoder_num_attention_heads = 16
    decoder_num_hidden_layers = 28
    layer_norm_eps = 1e-12
    patch_size = 16
    num_channels = 3
    num_patches = 16 * 16

    def __init__(self):
        super().__init__()
        self.decoder_embed = nn.Linear(self.latent_dim, self.decoder_hidden_size, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.decoder_hidden_size),
            requires_grad=False,
        )
        self.decoder_layers = nn.ModuleList(
            [
                _DecoderLayer(
                    hidden_size=self.decoder_hidden_size,
                    intermediate_size=self.decoder_intermediate_size,
                    num_heads=self.decoder_num_attention_heads,
                    eps=self.layer_norm_eps,
                )
                for _ in range(self.decoder_num_hidden_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_hidden_size, eps=self.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            self.decoder_hidden_size,
            self.patch_size**2 * self.num_channels,
            bias=True,
        )
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "RAEv2GeneralDecoder":
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"RAEv2 released decoder checkpoint not found: {path}. "
                "Place general/dinov3l-k7/decoder.pt at this path."
            )
        decoder = cls()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            for key in ("state_dict", "decoder"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported RAEv2 decoder checkpoint type: {type(payload)}")
        state_dict = {}
        for key, value in payload.items():
            if not isinstance(value, torch.Tensor):
                continue
            clean_key = str(key)
            for prefix in ("module.decoder.", "decoder.", "module."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    break
            state_dict[clean_key] = value
        incompatible = decoder.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "RAEv2 decoder checkpoint does not match the released ViT-XL architecture: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        decoder.requires_grad_(False)
        return decoder.to(device=device, dtype=dtype).eval()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1:] != (self.latent_dim, 16, 16):
            raise ValueError(
                "RAEv2 decoder expects released single-camera DINO features [B,1024,16,16], "
                f"got {tuple(latent.shape)}."
            )
        tokens = latent.flatten(2).transpose(1, 2)
        hidden_states = self.decoder_embed(tokens)
        cls_token = self.trainable_cls_token.expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([cls_token, hidden_states], dim=1) + self.decoder_pos_embed
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states)
        patches = self.decoder_pred(self.decoder_norm(hidden_states))[:, 1:]
        patches = patches.reshape(
            latent.shape[0],
            16,
            16,
            self.patch_size,
            self.patch_size,
            self.num_channels,
        )
        pixels = torch.einsum("nhwpqc->nchpwq", patches)
        return pixels.reshape(latent.shape[0], self.num_channels, 256, 256)
