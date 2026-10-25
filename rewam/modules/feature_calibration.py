from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureCalibration(nn.Module):
    """Normalize selected layers, average them, and add the last layer's spatial mean."""

    def __init__(self, layers: Sequence[int] = (11, 13, 15, 17, 19, 21, 23)) -> None:
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        if not self.layers:
            raise ValueError("Feature calibration requires at least one layer.")

    @torch.no_grad()
    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.layers):
            raise ValueError("Feature calibration requires one feature grid per selected layer.")
        batch, channels, frames, height, width = features[0].shape
        layer_tokens = []
        for feature in features:
            if feature.shape != features[0].shape:
                raise ValueError("Feature calibration layers must have the same shape.")
            tokens = feature.permute(0, 2, 3, 4, 1).reshape(batch * frames, height * width, channels)
            layer_tokens.append(F.layer_norm(tokens, (channels,), eps=1e-5))
        tokens = torch.stack(layer_tokens, dim=0).mean(dim=0)
        tokens = tokens + layer_tokens[-1].mean(dim=1, keepdim=True)
        return tokens.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()
