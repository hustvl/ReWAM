import json
from collections import defaultdict
from typing import Annotated, Any, Literal

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

ConstConstStr = Annotated[
    str,
    "format: 'const_min/const_max', where const_min and const_max give the constant range",
]
NormMode = Literal["min/max", "q01/q99", "z-score"] | ConstConstStr


class LinearNormalizer:
    """Normalize native action/state tensors using existing FastWAM statistics."""

    def __init__(
        self,
        shape_meta,
        default_mode: NormMode,
        stats,
        *,
        exception_mode: dict[str, dict[str, NormMode]] | None = None,
    ):
        self.normalizers = {}
        self.field_keys = {}
        self.stats = stats
        for kind in ("action", "state"):
            metadata = shape_meta[kind]
            if len(metadata) != 1:
                raise ValueError(f"ReWAM requires one {kind} vector.")
            key = metadata[0]["key"]
            self.field_keys[kind] = key
            field_stats = {
                name.removeprefix("global_"): value
                for name, value in stats[kind][key].items()
                if name.startswith("global_")
            }
            mode = (exception_mode or {}).get(kind, {}).get(key, default_mode)
            self.normalizers[kind] = SingleFieldLinearNormalizer(stats=field_stats, mode=mode)

    def get_stats(self):
        # The serialized statistics retain the original field keys; samples do not.
        return {kind: {self.field_keys[kind]: norm.get_stats()} for kind, norm in self.normalizers.items()}

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for kind, norm in self.normalizers.items():
            if kind in data:
                data[kind] = norm.forward(data[kind])
        return data

    def backward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for kind, norm in self.normalizers.items():
            if kind in data:
                data[kind] = norm.backward(data[kind])
        return data


class SingleFieldLinearNormalizer:
    std_reg = 1e-8
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0

    def __init__(self, stats, mode: NormMode = "min/max"):
        self.stats = stats
        self.mode = mode

        if mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            scale = 1.0 / (input_std + self.std_reg)
            offset = -input_mean / (input_std + self.std_reg)
        else:
            if mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif mode == "q01/q99":
                input_min, input_max = stats["q01"], stats["q99"]
            else:
                # parse const_min/const_max
                input_min, input_max = map(float, mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < self.range_tol
            input_range[ignore_dim] = self.output_max - self.output_min
            scale = (self.output_max - self.output_min) / input_range
            offset = self.output_min - scale * input_min
            offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]

        self.scale = scale
        self.offset = offset

    def get_stats(self):
        return self.stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.offset
        x = torch.clamp(x, -5.0, 5.0)
        return x

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.offset) / self.scale
        return x


def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):

    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable_stats = convert_tensor(dataset_stats)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)


def load_dataset_stats_from_json(file_path: str, try_convert_tensor: bool = True) -> dict[str, Any]:

    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True
            first = obj[0]
            if isinstance(first, (int, float)):
                return all(isinstance(x, (int, float)) for x in obj)
            elif isinstance(first, list):
                return all(is_numeric_list(item) for item in obj)
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr).to(torch.float32)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    return data


def split_robotwin_video(video: torch.Tensor) -> list[torch.Tensor]:
    """Split the top/front and bottom/wrist views from a [B,C,T,H,W] video."""
    if video.ndim != 5:
        raise ValueError(f"Camera video must be [B,C,T,H,W], got {tuple(video.shape)}.")
    height, width = video.shape[-2:]
    top_h = round(height * 2.0 / 3.0)
    if top_h <= 0 or top_h >= height or width % 2:
        raise ValueError(f"Invalid Robotwin camera layout size: {height}x{width}.")
    return [
        video[..., :top_h, :],
        video[..., top_h:, : width // 2],
        video[..., top_h:, width // 2 :],
    ]


def compose_camera_video(video: torch.Tensor, layout: str | None) -> torch.Tensor:
    """Compose [views, T, C, H, W]; output size follows the camera layout."""
    num_cameras = video.shape[0]
    if layout == "robotwin":
        if num_cameras != 3:
            raise ValueError(
                f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
            )
        cam_top = transforms_F.resize(
            video[0],
            size=[256, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )  # [T_video, C, 256, 320]
        cam_left = transforms_F.resize(
            video[1],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )  # [T_video, C, 128, 160]
        cam_right = transforms_F.resize(
            video[2],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )  # [T_video, C, 128, 160]
        bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
        video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
    elif num_cameras > 1:
        if layout == "horizontal":
            video = torch.cat(
                [video[i] for i in range(num_cameras)], dim=-1
            )  # [T_video, C, H, num_cameras*W]
        elif layout == "vertical":
            video = torch.cat(
                [video[i] for i in range(num_cameras)], dim=-2
            )  # [T_video, C, num_cameras*H, W]
        else:
            raise ValueError(
                f"Invalid concat_multi_camera: {layout}. " "Expected one of: horizontal, vertical, robotwin."
            )
    else:
        video = video[0]  # [T_video, C, H, W]
    return video
