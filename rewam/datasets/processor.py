from typing import Any

import numpy as np
import torch

from .utils import LinearNormalizer, NormMode


class ReWAMProcessor:
    def __init__(
        self,
        shape_meta: dict[str, Any],
        norm_default_mode: NormMode,
        transforms: dict[str, list[Any]] | list[Any] | None,
        drop_high_level_prob: float = 1.0,
        use_zh_instruction: bool = False,
        tokenizer: Any | None = None,
        norm_exception_mode: dict[str, dict[str, NormMode]] | None = None,
    ):
        self.shape_meta = shape_meta

        self.drop_high_level_prob = drop_high_level_prob
        self.use_zh_instruction = use_zh_instruction

        self.transforms = transforms

        self._is_train = None

        for kind in ("action", "state"):
            metadata = self.shape_meta[kind]
            if len(metadata) != 1:
                raise ValueError(f"ReWAM requires one {kind} vector, got {len(metadata)} fields.")
        self.action_dim = int(self.shape_meta["action"][0]["shape"])
        self.proprio_dim = int(self.shape_meta["state"][0]["shape"])

        self.norm_default_mode = norm_default_mode
        self.norm_exception_mode = norm_exception_mode
        self._normalizer = None

        self.tokenizer = tokenizer

    @property
    def is_train(self):
        if self._is_train is None:
            raise ValueError("is_train has not been set. Please call train() and eval() first.")
        return self._is_train

    @property
    def normalizer(self) -> LinearNormalizer:
        if self._normalizer is None:
            raise ValueError("normalizer has not been set. Please call set_normalizer_from_stats() first.")
        return self._normalizer

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def set_normalizer_from_stats(self, dataset_stats: dict[str, Any] | None = None):
        self._normalizer = LinearNormalizer(
            shape_meta=self.shape_meta,
            default_mode=self.norm_default_mode,
            exception_mode=self.norm_exception_mode,
            stats=dataset_stats,
        )

    def augment_instruction(self, data: dict[str, str]) -> str:
        """Choose and combine the sample's high/low-level instruction strings."""
        if "coarse_task" in data:
            high_level_instruction = data["coarse_task"]
        else:
            high_level_instruction = ""
        if "task" not in data:
            return f"[high] {high_level_instruction}"

        low_level_instruction = data["task"]
        # Galaxea lerobot use @ to split Chinese and English instruction
        if "@" in low_level_instruction:
            zh, eng = low_level_instruction.split("@")
            low_level_instruction = zh if self.use_zh_instruction else eng

        if np.random.rand() < self.drop_high_level_prob:
            instruction = f"{low_level_instruction}"
        else:
            instruction = f"[High]: {high_level_instruction}, [Low]: {low_level_instruction}"

        return instruction

    def validate_action_state(self, data: dict[str, Any]) -> dict[str, Any]:
        """Check native action/state tensors without wrapping or transforming them."""
        for kind, dimension in (
            ("action", self.action_dim),
            ("state", self.proprio_dim),
        ):
            if kind not in data:
                continue
            tensor = data[kind]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tensor.shape[-1] != dimension:
                raise ValueError(f"ReWAM {kind} must be a tensor with final dimension {dimension}.")
        return data

    def preprocess(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess the data for the policy model.

        Args:
            data: LeRobot sample with temporal tensors:
                - "action": optional tensor [action_horizon, action_dim]
                - "state": tensor [T_state, state_dim]
                - "images": Dict[str, torch.Tensor] -> [T_image, C, H, W]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "state_is_pad": torch.Tensor -> [T_state,]
                - "image_is_pad": torch.Tensor -> [T_image,]
                - "idx": int, sample index

        Returns:
            A collatable sample containing:
                - "pixel_values": torch.Tensor -> [num_input_cameras, T, 3, H, W]
                - "image_is_pad": torch.Tensor -> [T_image,]
                - "proprio": torch.Tensor -> [T_state, proprio_dim]
                - "state_is_pad": torch.Tensor -> [T_state,]
                - "action": Optional, torch.Tensor -> [action_horizon, action_dim]
                - "action_is_pad": Optional, torch.Tensor -> [action_horizon,]
                - "gt_action": optional copy of the unnormalized action for open-loop evaluation
                - "idx": int, sample index
        """
        sample = {}
        sample["instruction"] = self.augment_instruction(data)
        sample["image_is_pad"] = data["image_is_pad"]

        num_image_frames = len(data["image_is_pad"])
        processed_images = []
        for meta in self.shape_meta["images"]:
            key = meta["key"]
            image = data["images"][key]  # [T_image, C, H, W]
            if image.ndim != 4:
                raise ValueError(f"Expected images [T,3,H,W], got {tuple(image.shape)}.")

            current_transforms = (
                self.transforms[key] if isinstance(self.transforms, dict) else self.transforms
            )
            for trans in current_transforms or ():
                image = trans(image)

            if image.ndim != 4 or image.shape[:2] != (num_image_frames, 3):
                raise ValueError(
                    f"Expected RGB images [T,3,H,W] with T={num_image_frames}, "
                    f"got {tuple(image.shape)} after transforms for key {key}"
                )

            processed_images.append(image)
        sample["pixel_values"] = torch.stack(processed_images, dim=0)  # [cameras, T, C, H, W]

        # Open-loop evaluation compares predictions against unnormalized actions.
        if not self.is_train and "action" in data:
            sample["gt_action"] = data["action"].clone()

        self.validate_action_state(data)
        data = self.normalizer.forward(data)

        if "action" in data:
            sample["action"] = data["action"]  # [action_horizon, action_dim]
            sample["action_is_pad"] = data["action_is_pad"]  # [action_horizon,]
            sample["action_dim_is_pad"] = torch.zeros(
                self.action_dim, dtype=torch.bool, device=data["action"].device
            )  # [action_dim,]
            assert sample["action"].shape[-1] == self.action_dim

        sample["proprio"] = data["state"]  # [T_state, proprio_dim]
        sample["proprio_is_pad"] = data["state_is_pad"]  # [T_state,]
        sample["proprio_dim_is_pad"] = torch.zeros(
            self.proprio_dim, dtype=torch.bool, device=data["state"].device
        )  # [proprio_dim,]
        assert sample["proprio"].shape[-1] == self.proprio_dim

        sample["idx"] = data["idx"]

        return sample
