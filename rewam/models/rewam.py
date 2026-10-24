from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from rewam.datasets.utils import split_robotwin_video
from rewam.models.action_dit import ActionDiT
from rewam.models.encoder import DINOEncoder
from rewam.models.feature_calibration import FeatureCalibration
from rewam.models.losses import compute_action_flow_loss
from rewam.models.mot import MoT, build_world_action_mot_mask
from rewam.models.reconstruction_probe import ReconstructionProbe
from rewam.models.scheduler import WanContinuousFlowMatchScheduler
from rewam.models.temporal_bottleneck import (
    CausalRunningNormalization,
    TemporalRepresentationBottleneck,
)
from rewam.models.wan22.wan_video_dit import WanVideoDiT
from rewam.utils.config import as_plain_dict, as_thw

logger = logging.getLogger(__name__)


class ReWAM(nn.Module):
    """Representation-Diffusion + Action MoT.

    ReWAM keeps FastWAM's MoT/action training pattern, but replaces the WAN/VAE
    video latent branch with an online frozen visual representation branch.
    """

    def __init__(
        self,
        mot: MoT,
        vision_encoder: nn.Module,
        feature_calibration: nn.Module,
        temporal_bottleneck: dict[str, Any],
        camera_layout: str = "robotwin",
        text_encoder=None,
        tokenizer=None,
        text_dim: int | None = None,
        proprio_dim: int | None = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        world_train_shift: float | str = 5.0,
        world_infer_shift: float | str = 5.0,
        world_shift_base_dim: int = 4096,
        world_shift_scope: str = "future_tokens",
        world_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_world: float = 1.0,
        loss_lambda_action: float = 1.0,
        reconstruction_probe: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.mot = mot
        self.vision_encoder = vision_encoder
        self.feature_calibration = feature_calibration
        if camera_layout != "robotwin":
            raise ValueError("ReWAM requires the three-camera Robotwin layout.")
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self._capture_representation_viz = False
        self._last_representation_viz = None

        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self._world_train_shift = world_train_shift
        self._world_infer_shift = world_infer_shift
        self._world_shift_base_dim = int(world_shift_base_dim)
        self._world_shift_scope = world_shift_scope
        if self._world_shift_base_dim <= 0:
            raise ValueError("world_scheduler.shift_base_dim must be positive.")
        if world_shift_scope not in {"future_tokens", "all_tokens"}:
            raise ValueError("world_scheduler.shift_scope must be future_tokens or all_tokens.")
        # Automatic shifts are resolved once the actual latent shape is known.
        self.train_world_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=world_num_train_timesteps,
            shift=1.0 if world_train_shift == "auto" else float(world_train_shift),
        )
        self.infer_world_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=world_num_train_timesteps,
            shift=1.0 if world_infer_shift == "auto" else float(world_infer_shift),
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        bottleneck_cfg = as_plain_dict(temporal_bottleneck, required=True)
        self.target_dim = int(bottleneck_cfg["output_dim"])
        self.vision_encoder_output_dim = int(self.vision_encoder.output_dim)
        self.trb_gradient_mode = str(bottleneck_cfg.get("gradient_mode", "action_only")).lower()
        if self.trb_gradient_mode != "action_only":
            raise ValueError("The released ReWAM recipe requires action_only AGRS.")
        self.trb = TemporalRepresentationBottleneck(
            input_dim=self.vision_encoder_output_dim,
            output_dim=self.target_dim,
            patch_size=as_thw(bottleneck_cfg.get("patch_size", (2, 2, 2))),
            qk_dim=int(bottleneck_cfg.get("qk_dim", 512)),
            num_heads=int(bottleneck_cfg.get("num_heads", 8)),
            norm_eps=float(bottleneck_cfg.get("norm_eps", 1e-6)),
            rope_base=float(bottleneck_cfg.get("rope_base", 10000.0)),
            position_scale=float(bottleneck_cfg.get("position_scale", 16.0)),
        ).to(dtype=torch_dtype)
        normalization = as_plain_dict(bottleneck_cfg.get("causal_normalization"))
        self.causal_norm = CausalRunningNormalization(
            self.target_dim,
            eps=float(normalization.get("eps", 1e-5)),
            momentum=float(normalization.get("momentum", 0.1)),
        )
        self.reconstruction_probe = None
        self.reconstruction_probe_loss_weight = 0.0
        if reconstruction_probe is not None:
            reconstruction_cfg = as_plain_dict(reconstruction_probe)
            self.reconstruction_probe = ReconstructionProbe(
                input_dim=self.target_dim,
                output_dim=self.vision_encoder_output_dim,
                patch_size=self.trb.patch_size,
                decoder_path=reconstruction_cfg.get("decoder_path"),
            ).to(dtype=torch_dtype)
            self.reconstruction_probe_loss_weight = float(reconstruction_cfg.get("loss_weight", 1.0))
            if self.reconstruction_probe_loss_weight < 0:
                raise ValueError("reconstruction_probe.loss_weight must be non-negative.")
        if int(self.world_expert.in_dim) != self.target_dim:
            raise ValueError("World expert input dimension must match temporal_bottleneck.output_dim.")
        self.num_history_frames = self.trb.patch_size[0] - 1
        logger.info(
            "ReWAM TRB: input_dim=%d output_dim=%d patch_size=%s gradient_mode=%s",
            self.vision_encoder_output_dim,
            self.target_dim,
            self.trb.patch_size,
            self.trb_gradient_mode,
        )

        self.loss_lambda_world = float(loss_lambda_world)
        self.loss_lambda_action = float(loss_lambda_action)

        # Record the target dtype on the lazily loaded DINO backbone as well as
        # moving registered modules. TRB pretraining uses the same BF16 DINO
        # path, so leaving the WAM backbone at implicit FP32 would shift targets.
        self.to(device=device, dtype=torch_dtype)

    @classmethod
    def from_config(
        cls,
        *,
        world_dit_config: dict[str, Any],
        vision_encoder: dict[str, Any],
        feature_calibration: dict[str, Any],
        temporal_bottleneck: dict[str, Any],
        action_dit_config: dict[str, Any],
        camera_layout: str = "robotwin",
        mot_checkpoint_mixed_attn: bool = True,
        reconstruction_probe: dict[str, Any] | None = None,
        text_encoder=None,
        tokenizer=None,
        text_dim: int | None = None,
        proprio_dim: int | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        world_train_shift: float | str = 5.0,
        world_infer_shift: float | str = 5.0,
        world_shift_base_dim: int = 4096,
        world_shift_scope: str = "future_tokens",
        world_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_world: float = 1.0,
        loss_lambda_action: float = 1.0,
    ) -> ReWAM:
        vision_encoder = DINOEncoder(**as_plain_dict(vision_encoder))
        feature_calibration = FeatureCalibration(**as_plain_dict(feature_calibration))
        world_expert = WanVideoDiT(**world_dit_config).to(device=device, dtype=torch_dtype)
        action_expert = ActionDiT(**action_dit_config).to(device=device, dtype=torch_dtype)
        mot = MoT(
            mixtures={"video": world_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            mot=mot,
            vision_encoder=vision_encoder,
            feature_calibration=feature_calibration,
            camera_layout=camera_layout,
            temporal_bottleneck=temporal_bottleneck,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            world_train_shift=world_train_shift,
            world_infer_shift=world_infer_shift,
            world_shift_base_dim=world_shift_base_dim,
            world_shift_scope=world_shift_scope,
            world_num_train_timesteps=world_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_world=loss_lambda_world,
            loss_lambda_action=loss_lambda_action,
            reconstruction_probe=reconstruction_probe,
        )
        model.model_paths = {
            "world_dit": "RANDOM_INIT",
            "action_dit": "RANDOM_INIT",
        }
        return model

    @property
    def world_expert(self) -> WanVideoDiT:
        return self.mot.mixtures["video"]

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.mixtures["action"]

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def torch_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def set_representation_visualization_capture(self, enabled: bool):
        self._capture_representation_viz = bool(enabled)
        if not enabled:
            self._last_representation_viz = None

    def pop_last_representation_visualization(self):
        payload = self._last_representation_viz
        self._last_representation_viz = None
        return payload

    @staticmethod
    def _repr_latents_to_pca_rgb_strip(pred_repr: torch.Tensor, target_repr: torch.Tensor) -> torch.Tensor:
        if pred_repr.ndim != 5 or target_repr.ndim != 5:
            raise ValueError(
                "ReWAM representation visualization expects [B,C,T,H,W], "
                f"got {tuple(pred_repr.shape)} and {tuple(target_repr.shape)}."
            )
        if pred_repr.shape != target_repr.shape:
            raise ValueError(
                "ReWAM representation visualization shape mismatch: "
                f"pred={tuple(pred_repr.shape)} target={tuple(target_repr.shape)}."
            )

        _, channels, frames, height, width = [int(v) for v in pred_repr.shape]
        if channels <= 0 or frames <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"Invalid ReWAM representation visualization shape: {tuple(pred_repr.shape)}")

        pred = pred_repr[0].detach().float().cpu()
        target = target_repr[0].detach().float().cpu()
        pred_tokens = pred.permute(1, 2, 3, 0).reshape(frames * height * width, channels)
        target_tokens = target.permute(1, 2, 3, 0).reshape(frames * height * width, channels)
        combined = torch.cat([target_tokens, pred_tokens], dim=0)
        combined = combined - combined.mean(dim=0, keepdim=True)
        q = min(3, int(combined.shape[0] - 1), int(combined.shape[1]))
        if q <= 0:
            raise ValueError(
                f"Cannot PCA-project ReWAM representation tokens with shape {tuple(combined.shape)}."
            )
        _, _, v = torch.pca_lowrank(combined, q=q, center=False)
        projected = combined @ v[:, :q]
        if q < 3:
            projected = torch.cat(
                [projected, projected.new_zeros(projected.shape[0], 3 - q)],
                dim=1,
            )

        lo = projected.amin(dim=0, keepdim=True)
        hi = projected.amax(dim=0, keepdim=True)
        projected = ((projected - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
        token_count = frames * height * width
        target_rgb, pred_rgb = projected[:token_count], projected[token_count:]
        target_rgb = target_rgb.reshape(frames, height, width, 3).permute(0, 3, 1, 2)
        pred_rgb = pred_rgb.reshape(frames, height, width, 3).permute(0, 3, 1, 2)
        diff_rgb = (pred_rgb - target_rgb).abs()

        rows = []
        for frame_idx in range(frames):
            rows.append(
                torch.cat(
                    [target_rgb[frame_idx], pred_rgb[frame_idx], diff_rgb[frame_idx]],
                    dim=1,
                )
            )
        return torch.cat(rows, dim=2).contiguous()

    def _maybe_store_representation_visualization(self, pred_repr: torch.Tensor, target_repr: torch.Tensor):
        if not self._capture_representation_viz:
            return
        self._capture_representation_viz = False
        self._last_representation_viz = None
        with torch.no_grad():
            image = self._repr_latents_to_pca_rgb_strip(
                pred_repr=pred_repr,
                target_repr=target_repr,
            )
        self._last_representation_viz = {
            "tag": "train/representation_pca/ra_target_pred_diff",
            "image": image,
        }

    @torch.no_grad()
    def encode_prompt(self, prompt: str | Sequence[str]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _merge_camera_trb_latents(self, camera_latents: list[torch.Tensor]) -> torch.Tensor:
        if len(camera_latents) != 3:
            raise ValueError("ReWAM TRB requires front, left-wrist, and right-wrist states.")
        front, left, right = camera_latents
        if any(latent.ndim != 5 or latent.shape[:3] != front.shape[:3] for latent in camera_latents):
            raise ValueError("ReWAM camera states must share [B,C,T].")
        if left.shape[-2:] != right.shape[-2:]:
            raise ValueError("ReWAM wrist state grids must match.")
        if front.shape[-2] != 2 * left.shape[-2] or front.shape[-1] != left.shape[-1] + right.shape[-1]:
            raise ValueError("ReWAM front grid must match the two-wrist Robotwin layout.")
        return torch.cat([front, torch.cat([left, right], dim=-1)], dim=-2).contiguous()

    @torch.no_grad()
    def _encode_camera_features(
        self,
        camera_videos: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if not camera_videos:
            raise ValueError("Camera feature encoding received no camera videos.")
        batch_size = int(camera_videos[0].shape[0])
        resolution_groups: dict[tuple[int, int], list[int]] = {}
        for camera_idx, camera_video in enumerate(camera_videos):
            if camera_video.ndim != 5 or int(camera_video.shape[0]) != batch_size:
                raise ValueError(f"Invalid camera video {camera_idx}: {tuple(camera_video.shape)}.")
            resolution = (
                int(camera_video.shape[-2]),
                int(camera_video.shape[-1]),
            )
            resolution_groups.setdefault(resolution, []).append(camera_idx)

        camera_features: list[torch.Tensor | None] = [None] * len(camera_videos)
        for camera_indices in resolution_groups.values():
            packed_video = torch.cat(
                [camera_videos[idx] for idx in camera_indices],
                dim=0,
            )
            features = self.feature_calibration(
                self.vision_encoder(packed_video, layers=self.feature_calibration.layers)
            )
            if int(features.shape[1]) != self.vision_encoder_output_dim:
                raise ValueError(
                    "Vision encoder feature dim mismatch before trb: "
                    f"got {features.shape[1]}, expected {self.vision_encoder_output_dim}."
                )
            for camera_idx, feature in zip(
                camera_indices,
                features.split(batch_size, dim=0),
            ):
                camera_features[camera_idx] = feature.to(
                    device=self.device,
                    dtype=self.torch_dtype,
                )

        if any(feature is None for feature in camera_features):
            raise RuntimeError("Failed to produce a DINO feature target for every camera.")
        return [feature for feature in camera_features if feature is not None]

    def _encode_trb_representation(
        self,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        camera_videos = split_robotwin_video(video)
        camera_features = self._encode_camera_features(camera_videos)
        camera_latents = self.trb(camera_features)
        latents = self._merge_camera_trb_latents(camera_latents)
        return latents.to(device=self.device, dtype=self.torch_dtype), camera_features

    def _normalize_trb_latents(
        self,
        raw_latents: torch.Tensor,
        *,
        update_stats: bool | None = None,
    ) -> torch.Tensor:
        return self.causal_norm(raw_latents, update_stats=update_stats)

    def _denormalize_trb_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.causal_norm.denormalize(latents)

    @torch.no_grad()
    def _encode_representation_latents(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B,C,T,H,W], got {tuple(video.shape)}")
        raw_latents, _ = self._encode_trb_representation(video)
        return self._normalize_trb_latents(raw_latents, update_stats=False)

    def _validate_representation_temporal_size(
        self, features: torch.Tensor, target_frames: int
    ) -> torch.Tensor:
        if int(features.shape[2]) != int(target_frames):
            raise ValueError(f"ReWAM temporal mismatch: got {features.shape[2]}, expected {target_frames}.")
        return features

    def _num_representation_steps(self, num_video_frames: int) -> int:
        patch_t = self.trb.patch_size[0]
        if num_video_frames <= 0 or num_video_frames % patch_t:
            raise ValueError(
                f"Sampled image frame count must be positive and divisible by TRB patch_size[0]={patch_t}."
            )
        return num_video_frames // patch_t

    def _prepare_training_video(self, video: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_steps = self._num_representation_steps(video.shape[2])
        if num_steps < 2:
            raise ValueError("ReWAM training requires current and future representation steps.")
        return video, num_steps

    def _select_image_pad_mask(self, image_is_pad: torch.Tensor) -> torch.Tensor:
        num_steps = self._num_representation_steps(image_is_pad.shape[1])
        return image_is_pad.reshape(image_is_pad.shape[0], num_steps, self.trb.patch_size[0]).all(dim=-1)

    def _resolve_world_shift(self, value: float | str, latents: torch.Tensor) -> float:
        if value != "auto":
            return float(value)
        _, channels, frames, height, width = latents.shape
        if self._world_shift_scope == "future_tokens":
            frames -= 1
        return math.sqrt(channels * frames * height * width / self._world_shift_base_dim)

    def _prepare_context(
        self,
        *,
        prompt: str | Sequence[str] | None,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        proprio: torch.Tensor | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if context.shape[0] != batch_size:
            if context.shape[0] == 1 and batch_size > 1:
                context = context.expand(batch_size, -1, -1)
                context_mask = context_mask.expand(batch_size, -1)
            else:
                raise ValueError(f"Context batch mismatch: got {context.shape[0]}, expected {batch_size}.")
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        return context, context_mask

    def _prepare_trb_latents_for_training(
        self,
        online_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if int(online_latents.shape[2]) < 2:
            raise ValueError("Trainable representation trb requires current and future latents.")

        online_current = online_latents[:, :, :1]
        detached_future = online_latents[:, :, 1:].detach()
        current_proxy = online_current.detach()
        if torch.is_grad_enabled():
            current_proxy = current_proxy.requires_grad_(True)
        return torch.cat((current_proxy, detached_future), dim=2), current_proxy

    def build_inputs(self, sample):
        video = sample["video"]
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got {video.shape[1]}")
        batch_size, _, num_frames, _, _ = video.shape
        if num_frames <= 1:
            raise ValueError(f"`sample['video']` must contain at least 2 frames, got {num_frames}.")
        video, expected_repr_steps = self._prepare_training_video(video)
        if "action" not in sample:
            raise ValueError("`sample['action']` is required for ReWAM training.")
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}")

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        raw_trb_latents, trb_feature_targets = self._encode_trb_representation(input_video)
        raw_trb_latents = self._validate_representation_temporal_size(
            raw_trb_latents,
            expected_repr_steps,
        )
        # The ReWAM wrapper stays in eval mode while selected trainable
        # submodules run in train mode. Let the normalization module's own
        # state decide whether lagged statistics should update.
        online_representation_latents = self._normalize_trb_latents(raw_trb_latents)
        representation_latents, trb_current_proxy = self._prepare_trb_latents_for_training(
            online_representation_latents
        )
        num_repr_steps = int(representation_latents.shape[2])
        if num_repr_steps <= 1:
            raise ValueError(
                f"ReWAM representation latents must contain at least 2 steps, got {num_repr_steps}."
            )
        first_frame_latents = representation_latents[:, :, 0:1]

        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if (context is None) != (context_mask is None):
            raise ValueError("`context` and `context_mask` must be provided together.")
        if context is None or context_mask is None:
            prompt = sample.get("prompt")
        else:
            prompt = None
        proprio_seq = sample.get("proprio", None)
        if self.proprio_encoder is not None:
            if proprio_seq is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio_seq.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be [B,T,D], got {tuple(proprio_seq.shape)}")
            proprio = proprio_seq[:, 0, :]
        else:
            proprio = None
        context, context_mask = self._prepare_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=(proprio.to(device=self.device, dtype=self.torch_dtype) if proprio is not None else None),
            batch_size=batch_size,
        )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.shape != (batch_size, num_frames):
                raise ValueError("image_is_pad must match the sampled video's batch and frame dimensions.")
            image_is_pad = self._select_image_pad_mask(image_is_pad)
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "video": input_video,
            "representation_latents": representation_latents,
            "online_representation_latents": online_representation_latents,
            "raw_trb_latents": raw_trb_latents,
            "trb_current_proxy": trb_current_proxy,
            "trb_feature_targets": trb_feature_targets,
            "first_frame_latents": first_frame_latents,
            "action": action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _encode_inference_first_frame_latents(
        self,
        input_image: torch.Tensor,
        history_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # History is ordered oldest to newest: [B,C,T_history,H,W].
        current = input_image.unsqueeze(2)
        if history_images is None:
            history_images = current.expand(-1, -1, self.num_history_frames, -1, -1)
        expected = (*input_image.shape[:2], self.num_history_frames, *input_image.shape[2:])
        if tuple(history_images.shape) != expected:
            raise ValueError(f"history_images must be {expected}, got {tuple(history_images.shape)}.")
        input_video = torch.cat((history_images.to(input_image), current), dim=2)
        latents = self._encode_representation_latents(input_video)
        return self._validate_representation_temporal_size(latents, 1)

    def validate_inference_timeline(
        self,
        *,
        num_video_frames: int,
        action_horizon: int,
    ) -> int:
        """Validate source-frame and action horizons against ReWAM's latent timeline."""
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")
        num_repr_steps = self._num_representation_steps(num_video_frames)
        num_future_repr_steps = num_repr_steps - 1
        if num_future_repr_steps <= 0:
            raise ValueError(
                "ReWAM inference requires at least one future representation step, "
                f"got num_repr_steps={num_repr_steps}."
            )
        return num_repr_steps

    def _split_representation_camera_latents(self, latents: torch.Tensor) -> list[torch.Tensor]:
        if latents.ndim != 5:
            raise ValueError("ReWAM states must be [B,C,T,H,W].")
        height, width = latents.shape[-2:]
        front_height = round(height * 2.0 / 3.0)
        wrist_width = width // 2
        if not 0 < front_height < height or not 0 < wrist_width < width:
            raise ValueError("ReWAM state grid is too small for the Robotwin layout.")
        return [
            latents[..., :front_height, :],
            latents[..., front_height:, :wrist_width],
            latents[..., front_height:, wrist_width:],
        ]

    def _decode_trb_feature_latents(self, latents: torch.Tensor) -> list[torch.Tensor]:
        if self.reconstruction_probe is None:
            raise ValueError(
                "Feature reconstruction is unavailable; configure reconstruction_probe for RGB diagnostics."
            )
        camera_latents = self._split_representation_camera_latents(latents)
        batch_size = int(latents.shape[0])
        resolution_groups: dict[tuple[int, int], list[int]] = {}
        for camera_idx, camera_latent in enumerate(camera_latents):
            resolution = (int(camera_latent.shape[-2]), int(camera_latent.shape[-1]))
            resolution_groups.setdefault(resolution, []).append(camera_idx)

        decoded_features: list[torch.Tensor | None] = [None] * len(camera_latents)
        for camera_indices in resolution_groups.values():
            packed_latents = torch.cat([camera_latents[idx] for idx in camera_indices], dim=0)
            packed_features = self.reconstruction_probe(packed_latents)
            chunks = packed_features.split(batch_size, dim=0)
            for camera_idx, decoded in zip(camera_indices, chunks):
                decoded_features[camera_idx] = decoded
        if any(features is None for features in decoded_features):
            raise RuntimeError("Failed to decode every camera trb latent to DINO features.")
        return [features for features in decoded_features if features is not None]

    def _reconstruction_probe_training_loss(
        self,
        *,
        representation_latents: torch.Tensor,
        feature_targets: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.reconstruction_probe is None or self.reconstruction_probe_loss_weight == 0.0:
            return representation_latents.new_zeros(())
        decoded_features = self._decode_trb_feature_latents(representation_latents.detach())
        if len(decoded_features) != len(feature_targets):
            raise ValueError(
                "TRB feature decoder camera count mismatch: "
                f"decoded={len(decoded_features)} target={len(feature_targets)}."
            )
        losses = []
        for camera_idx, (decoded, target) in enumerate(zip(decoded_features, feature_targets)):
            if decoded.shape != target.shape:
                raise ValueError(
                    f"TRB feature decoder camera {camera_idx} shape mismatch: "
                    f"decoded={tuple(decoded.shape)} target={tuple(target.shape)}."
                )
            losses.append(F.mse_loss(decoded.float(), target.detach().float()))
        return torch.stack(losses).mean()

    def _strip_history_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        if len(frames) <= self.num_history_frames:
            raise ValueError("ReWAM reconstruction must include history and current frames.")
        return frames[self.num_history_frames :]

    @torch.no_grad()
    def reconstruct_representation_video(self, video: torch.Tensor) -> list[Image.Image]:
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or int(video.shape[0]) != 1:
            raise ValueError(f"Representation reconstruction expects [1,3,T,H,W], got {tuple(video.shape)}.")
        selected_video, _ = self._prepare_training_video(video)
        selected_video = selected_video.to(device=self.device, dtype=self.torch_dtype)
        latents = self._encode_representation_latents(selected_video)
        return self._strip_history_frames(self._decode_representation_latents(latents))

    @torch.no_grad()
    def prepare_representation_rgb_target(self, video: torch.Tensor) -> list[Image.Image]:
        """Match GT camera layout and resolution to the released RAE decoder output."""
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5 or int(video.shape[0]) != 1:
            raise ValueError(f"Representation RGB target expects [1,3,T,H,W], got {tuple(video.shape)}.")
        selected_video, _ = self._prepare_training_video(video)
        camera_images = []
        for camera_video in split_robotwin_video(selected_video):
            batch, _, frames = camera_video.shape[:3]
            images = camera_video.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, 3, camera_video.shape[-2], camera_video.shape[-1]
            )
            images = ((images.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
            camera_images.append(F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False))
        target = self._merge_square_camera_rgb(camera_images, batch=batch, frames=frames)
        return self._strip_history_frames(self._video_tensor_to_pil(target))

    @staticmethod
    def _merge_square_camera_rgb(
        camera_images: list[torch.Tensor],
        *,
        batch: int,
        frames: int,
    ) -> torch.Tensor:
        if not camera_images:
            raise ValueError("No camera RGB images were provided.")
        camera_videos = []
        for images in camera_images:
            if images.ndim != 4 or int(images.shape[0]) != batch * frames:
                raise ValueError(f"Camera RGB images must be [B*T,3,H,W], got {tuple(images.shape)}.")
            camera_videos.append(
                images.reshape(batch, frames, 3, images.shape[-2], images.shape[-1]).permute(0, 2, 1, 3, 4)
            )
        return torch.cat(camera_videos, dim=-1).contiguous()

    @staticmethod
    def _video_tensor_to_pil(video: torch.Tensor) -> list[Image.Image]:
        if video.ndim != 5 or int(video.shape[0]) != 1 or int(video.shape[1]) != 3:
            raise ValueError(f"Decoded RGB video must be [1,3,T,H,W], got {tuple(video.shape)}.")
        frames = video[0].permute(1, 2, 3, 0).mul(255.0).round().byte().cpu().numpy()
        return [Image.fromarray(frame, mode="RGB") for frame in frames]

    @torch.no_grad()
    def _decode_camera_features_to_rgb_tensor(
        self,
        camera_features: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.reconstruction_probe is None:
            raise ValueError("RGB diagnostics require reconstruction_probe.")
        decoded_cameras = []
        batch = frames = None
        for features in camera_features:
            camera_batch, _, camera_frames = features.shape[:3]
            if batch is None:
                batch, frames = int(camera_batch), int(camera_frames)
            elif (int(camera_batch), int(camera_frames)) != (batch, frames):
                raise ValueError("All camera feature tensors must share batch and temporal dimensions.")
            decoded_cameras.append(self.reconstruction_probe.decode_rgb(features))
        if batch is None or frames is None:
            raise ValueError("No camera DINO features were provided for RGB decoding.")
        return self._merge_square_camera_rgb(decoded_cameras, batch=batch, frames=frames)

    @torch.no_grad()
    def _decode_representation_latents(self, latents: torch.Tensor) -> list[Image.Image]:
        raw_latents = self._denormalize_trb_latents(latents)
        camera_features = self._decode_trb_feature_latents(raw_latents)
        decoded = self._decode_camera_features_to_rgb_tensor(camera_features)
        return self._video_tensor_to_pil(decoded)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        return build_world_action_mot_mask(
            world_expert=self.world_expert,
            world_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            world_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        *,
        latents_repr: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_repr: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repr_pre = self.world_expert.pre_dit(
            x=latents_repr,
            timestep=timestep_repr,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=int(repr_pre["tokens"].shape[1]),
            action_seq_len=int(action_pre["tokens"].shape[1]),
            video_tokens_per_frame=int(repr_pre["meta"]["tokens_per_frame"]),
            device=repr_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={"video": repr_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": repr_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {
                    "context": repr_pre["context"],
                    "mask": repr_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={"video": repr_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        return (
            self.world_expert.post_dit(tokens_out["video"], repr_pre),
            self.action_expert.post_dit(tokens_out["action"], action_pre),
        )

    def _sigma_from_timestep(self, timestep: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.train_world_scheduler.num_train_timesteps)).to(
            device=target.device,
            dtype=target.dtype,
        )
        if sigma.ndim == 0:
            return sigma
        if sigma.ndim == 1:
            if sigma.shape[0] not in (1, target.shape[0]):
                raise ValueError(
                    "Batch timestep shape mismatch: "
                    f"timestep={tuple(timestep.shape)}, target={tuple(target.shape)}."
                )
            return sigma.view(-1, *([1] * (target.ndim - 1)))
        if sigma.ndim == 2:
            if target.ndim < 3 or sigma.shape != (target.shape[0], target.shape[2]):
                raise ValueError(
                    "Per-frame timestep must match target [B, C, T, ...]: "
                    f"timestep={tuple(timestep.shape)}, target={tuple(target.shape)}."
                )
            return sigma.view(target.shape[0], 1, target.shape[2], *([1] * (target.ndim - 3)))
        raise ValueError(f"Unsupported timestep shape: {tuple(timestep.shape)}.")

    def _sample_representation_training_timestep(
        self,
        representation_latents: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(representation_latents.shape[0])
        num_frames = int(representation_latents.shape[2])
        if num_frames < 2:
            raise ValueError(
                "Per-frame representation timesteps require at least one observation "
                "and one future latent frame."
            )
        self.train_world_scheduler.set_shift(
            self._resolve_world_shift(self._world_train_shift, representation_latents)
        )
        future_timestep = self.train_world_scheduler.sample_training_t(
            batch_size=batch_size * (num_frames - 1),
            device=self.device,
            dtype=representation_latents.dtype,
        ).reshape(batch_size, num_frames - 1)
        clean_observation_timestep = torch.zeros(
            (batch_size, 1),
            device=future_timestep.device,
            dtype=future_timestep.dtype,
        )
        return torch.cat((clean_observation_timestep, future_timestep), dim=1)

    def _prediction_to_clean_state(
        self,
        *,
        pred_repr: torch.Tensor,
        noisy_repr: torch.Tensor,
        timestep_repr: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self._sigma_from_timestep(timestep_repr, pred_repr)
        return noisy_repr - sigma * pred_repr

    @staticmethod
    def _safe_scalar(value: torch.Tensor) -> float:
        return float(value.detach().float().mean().item())

    def _representation_monitor_metrics(
        self,
        *,
        representation_latents: torch.Tensor,
        clean_future: torch.Tensor,
        noisy_future: torch.Tensor,
        noise_future: torch.Tensor,
        pred_repr: torch.Tensor,
        timestep_repr: torch.Tensor,
        loss_world_raw: torch.Tensor,
        loss_world: torch.Tensor,
    ) -> dict[str, float]:
        future_timestep = timestep_repr[:, 1:] if timestep_repr.ndim == 2 else timestep_repr
        metrics = {
            "repr/target_mean": self._safe_scalar(clean_future.mean()),
            "repr/target_std": self._safe_scalar(clean_future.std()),
            "repr/target_norm": self._safe_scalar(clean_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/noise_norm": self._safe_scalar(noise_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/noisy_norm": self._safe_scalar(noisy_future.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/pred_norm": self._safe_scalar(pred_repr.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt()),
            "repr/loss_raw": float(loss_world_raw.detach().float().item()),
            "repr/loss_weighted": float(loss_world.detach().float().item()),
            "repr/sigma_mean": self._safe_scalar(
                future_timestep.float() / float(self.train_world_scheduler.num_train_timesteps)
            ),
            "repr/shift": float(self.train_world_scheduler.shift),
        }
        if timestep_repr.ndim == 2 and timestep_repr.shape[1] > 1:
            future_sigma = future_timestep.float() / float(self.train_world_scheduler.num_train_timesteps)
            metrics["repr/sigma_future_std"] = self._safe_scalar(future_sigma.std(dim=1, unbiased=False))
        if int(representation_latents.shape[2]) > 1:
            delta = representation_latents[:, :, 1:] - representation_latents[:, :, :-1]
            metrics["repr/delta_norm"] = self._safe_scalar(delta.float().pow(2).mean(dim=(1, 2, 3, 4)).sqrt())
        return metrics

    def training_loss(self, sample, tiled: bool = False):
        del tiled
        inputs = self.build_inputs(sample)
        representation_latents = inputs["representation_latents"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_repr = torch.randn_like(representation_latents)
        batch_size = int(representation_latents.shape[0])
        timestep_repr = self._sample_representation_training_timestep(representation_latents)
        noisy_repr = self.train_world_scheduler.add_noise(
            representation_latents,
            noise_repr,
            timestep_repr,
        )
        target_repr = self.train_world_scheduler.training_target(
            representation_latents, noise_repr, timestep_repr
        )
        # Keep the current state clean while noising the future states.
        noisy_repr[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        repr_pre = self.world_expert.pre_dit(
            x=noisy_repr,
            timestep=timestep_repr,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=repr_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(repr_pre["meta"]["tokens_per_frame"]),
            device=repr_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": repr_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": repr_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": repr_pre["context"],
                    "mask": repr_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": repr_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_repr = self.world_expert.post_dit(tokens_out["video"], repr_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        pred_repr = pred_repr[:, :, 1:]
        target_repr = target_repr[:, :, 1:]
        clean_future = representation_latents[:, :, 1:]
        noisy_future = noisy_repr[:, :, 1:]
        noise_future = noise_repr[:, :, 1:]
        if self._capture_representation_viz:
            future_timestep_repr = timestep_repr[:, 1:] if timestep_repr.ndim == 2 else timestep_repr
            clean_pred_for_viz = self._prediction_to_clean_state(
                pred_repr=pred_repr,
                noisy_repr=noisy_future,
                timestep_repr=future_timestep_repr,
            )
            viz_pred = clean_pred_for_viz
            viz_target = clean_future
            self._maybe_store_representation_visualization(
                pred_repr=viz_pred,
                target_repr=viz_target,
            )
        loss_world_per_frame = F.mse_loss(pred_repr.float(), target_repr.float(), reduction="none").mean(
            dim=(1, 3, 4)
        )
        if image_is_pad is None:
            valid_repr = torch.ones_like(loss_world_per_frame)
        else:
            if image_is_pad.shape[1] != loss_world_per_frame.shape[1] + 1:
                raise ValueError(
                    "Representation-loss mask shape mismatch: "
                    f"mask steps={image_is_pad.shape[1]}, "
                    f"loss steps={loss_world_per_frame.shape[1]}."
                )
            valid_repr = (~image_is_pad[:, 1:]).to(
                device=loss_world_per_frame.device,
                dtype=loss_world_per_frame.dtype,
            )
        valid_count = valid_repr.sum(dim=1).clamp(min=1.0)
        loss_world_per_sample = (loss_world_per_frame * valid_repr).sum(dim=1) / valid_count
        repr_weight = self.train_world_scheduler.training_weight(timestep_repr[:, 1:]).to(
            loss_world_per_frame.device, dtype=loss_world_per_frame.dtype
        )
        weighted_loss_world_per_sample = (loss_world_per_frame * repr_weight * valid_repr).sum(
            dim=1
        ) / valid_count
        loss_world_raw = loss_world_per_sample.mean()
        loss_world = weighted_loss_world_per_sample.mean()

        loss_action = compute_action_flow_loss(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            timestep_action=timestep_action,
            action_scheduler=self.train_action_scheduler,
        )

        loss_trb_route = loss_action.new_zeros(())
        trb_action_proxy_grad_norm = None
        trb_current_proxy = inputs["trb_current_proxy"]
        if self.trb_gradient_mode == "action_only" and torch.is_grad_enabled():
            if trb_current_proxy is None:
                raise RuntimeError("Action-only trb routing is missing its current-latent proxy.")
            trb_action_gradient = torch.autograd.grad(
                self.loss_lambda_action * loss_action,
                trb_current_proxy,
                retain_graph=True,
                create_graph=False,
            )[0]
            online_current = inputs["online_representation_latents"][:, :, :1]
            loss_trb_route = ((online_current - online_current.detach()) * trb_action_gradient.detach()).sum()
            trb_action_proxy_grad_norm = self._safe_scalar(trb_action_gradient.float().pow(2).mean().sqrt())

        loss_decoder_raw = self._reconstruction_probe_training_loss(
            representation_latents=inputs["raw_trb_latents"],
            feature_targets=inputs["trb_feature_targets"],
        )
        loss_decoder = self.reconstruction_probe_loss_weight * loss_decoder_raw

        loss_total = (
            self.loss_lambda_world * loss_world
            + self.loss_lambda_action * loss_action
            + loss_decoder
            + loss_trb_route
        )
        loss_dict = {
            "loss_world": self.loss_lambda_world * float(loss_world.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_reconstruction_probe": float(loss_decoder.detach().item()),
            "repr/trb_condition_gradient": float(self.trb_gradient_mode == "condition_and_action"),
            "repr/trb_action_only_gradient": float(self.trb_gradient_mode == "action_only"),
            "repr/trb_dim": float(self.target_dim),
        }
        if trb_action_proxy_grad_norm is not None:
            loss_dict["repr/trb_action_proxy_grad_rms"] = trb_action_proxy_grad_norm
        trb_attention_entropy = getattr(
            self.trb,
            "last_attention_entropy",
            None,
        )
        if trb_attention_entropy is not None:
            loss_dict["repr/trb_attention_entropy"] = self._safe_scalar(trb_attention_entropy)
        raw_trb_latents = inputs["raw_trb_latents"]
        loss_dict["repr/trb_raw_mean"] = self._safe_scalar(raw_trb_latents.mean())
        loss_dict["repr/trb_raw_std"] = self._safe_scalar(raw_trb_latents.std())
        loss_dict["repr/trb_norm_running_mean_abs"] = self._safe_scalar(
            self.causal_norm.running_mean.abs().mean()
        )
        loss_dict["repr/trb_norm_running_std"] = self._safe_scalar(
            self.causal_norm.running_var.clamp_min(0.0).sqrt().mean()
        )
        loss_dict["repr/trb_norm_batches"] = float(self.causal_norm.num_batches_tracked.item())
        loss_dict.update(
            self._representation_monitor_metrics(
                representation_latents=representation_latents,
                clean_future=clean_future,
                noisy_future=noisy_future,
                noise_future=noise_future,
                pred_repr=pred_repr,
                timestep_repr=timestep_repr,
                loss_world_raw=loss_world_raw,
                loss_world=loss_world,
            )
        )
        return loss_total, loss_dict

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: str | None,
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        history_images: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del action, negative_prompt, text_cfg_scale, tiled
        self.eval()
        num_repr_steps = self.validate_inference_timeline(
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
        )
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        context, context_mask = self._prepare_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            batch_size=1,
        )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if history_images is not None:
            history_images = history_images.to(device=self.device, dtype=self.torch_dtype)
        first_frame_repr = self._encode_inference_first_frame_latents(
            input_image,
            history_images=history_images,
        )
        _, repr_dim, _, repr_h, repr_w = first_frame_repr.shape

        repr_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_repr = torch.randn(
            (1, repr_dim, num_repr_steps, repr_h, repr_w),
            generator=repr_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_repr[:, :, 0:1] = first_frame_repr

        self.infer_world_scheduler.set_shift(self._resolve_world_shift(self._world_infer_shift, latents_repr))
        infer_timesteps_repr, infer_deltas_repr = self.infer_world_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=latents_repr.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        if len(infer_timesteps_repr) != len(infer_timesteps_action):
            raise ValueError("Representation/action inference schedules must have the same number of steps.")
        for step_t_repr, step_delta_repr, step_t_action, step_delta_action in zip(
            infer_timesteps_repr,
            infer_deltas_repr,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            pred_repr, pred_action = self._predict_joint_noise(
                latents_repr=latents_repr,
                latents_action=latents_action,
                timestep_repr=step_t_repr.unsqueeze(0).to(device=self.device, dtype=latents_repr.dtype),
                timestep_action=step_t_action.unsqueeze(0).to(device=self.device, dtype=latents_action.dtype),
                context=context,
                context_mask=context_mask,
            )
            latents_repr = self.infer_world_scheduler.step(
                pred_repr,
                step_delta_repr,
                latents_repr,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action,
                step_delta_action,
                latents_action,
            )
            latents_repr[:, :, 0:1] = first_frame_repr

        decode_latents = latents_repr
        return {
            "video": self._strip_history_frames(self._decode_representation_latents(decode_latents)),
            "representation": decode_latents.detach().to(device="cpu", dtype=torch.float32),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def _predict_cached_action_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: dict[str, torch.Tensor],
        freqs: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        tokens, _t, t_mod = self.action_expert.prepare_tokens(latents, timestep)
        tokens = self.mot.forward_action_with_video_cache(
            action_tokens=tokens,
            action_freqs=freqs,
            action_t_mod=t_mod,
            action_context_payload=conditioning,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.head(tokens)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: str | None,
        input_image: torch.Tensor,
        action_horizon: int,
        history_images: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        num_video_frames: int | None = None,
        compile_action_infer: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale, tiled, num_video_frames
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        if action_horizon <= 0 or action_horizon > self.action_expert.freqs.shape[0]:
            raise ValueError(f"Invalid action horizon {action_horizon} for the action RoPE cache.")
        context, context_mask = self._prepare_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            batch_size=1,
        )

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if history_images is not None:
            history_images = history_images.to(device=self.device, dtype=self.torch_dtype)
        first_frame_repr = self._encode_inference_first_frame_latents(
            input_image,
            history_images=history_images,
        )
        timestep_repr = torch.zeros((1,), dtype=first_frame_repr.dtype, device=self.device)
        repr_pre = self.world_expert.pre_dit(
            x=first_frame_repr,
            timestep=timestep_repr,
            context=context,
            context_mask=context_mask,
        )
        video_seq_len = int(repr_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(repr_pre["meta"]["tokens_per_frame"]),
            device=repr_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=repr_pre["tokens"],
            video_freqs=repr_pre["freqs"],
            video_t_mod=repr_pre["t_mod"],
            video_context_payload={
                "context": repr_pre["context"],
                "mask": repr_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        conditioning, action_freqs = self.action_expert.prepare_conditioning(
            context, context_mask, latents_action.shape[1]
        )
        denoise = self._predict_cached_action_noise
        if compile_action_infer:
            if not hasattr(self, "_compiled_action_denoise"):
                self._compiled_action_denoise = torch.compile(
                    self._predict_cached_action_noise, mode="reduce-overhead", fullgraph=True
                )
            denoise = self._compiled_action_denoise

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            if compile_action_infer:
                torch.compiler.cudagraph_mark_step_begin()
            pred_action = denoise(
                latents_action,
                timestep_action,
                conditioning,
                action_freqs,
                video_kv_cache,
                attention_mask,
                video_seq_len,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            if compile_action_infer:
                # The next CUDA Graph replay may overwrite graph-owned outputs.
                latents_action = latents_action.clone()

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

    @torch.no_grad()
    def infer(
        self,
        prompt: str | None,
        input_image: torch.Tensor,
        history_images: torch.Tensor | None = None,
        num_frames: int | None = None,
        action: torch.Tensor | None = None,
        action_horizon: int | None = None,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
    ) -> dict[str, Any]:
        del action, action_cfg_scale
        if action_horizon is None:
            raise ValueError("ReWAM.infer requires `action_horizon` because it only produces action output.")
        return self.infer_action(
            prompt=prompt,
            input_image=input_image,
            history_images=history_images,
            action_horizon=int(action_horizon),
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            compile_action_infer=compile_action_infer,
            num_video_frames=num_frames,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "model_class": self.__class__.__name__,
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        payload["trb"] = self.trb.state_dict()
        payload["causal_norm"] = self.causal_norm.state_dict()
        if self.reconstruction_probe is not None:
            payload["reconstruction_probe"] = self.reconstruction_probe.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        """Restore all configured components strictly."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        components = {
            "mot": self.mot,
            "trb": self.trb,
            "causal_norm": self.causal_norm,
            "reconstruction_probe": self.reconstruction_probe,
            "proprio_encoder": self.proprio_encoder,
        }
        for name, module in components.items():
            if module is not None and name not in payload:
                raise ValueError(f"ReWAM checkpoint missing required component {name!r}.")
        for name, module in components.items():
            if module is not None:
                module.load_state_dict(payload[name], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
