from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ReWAMDefaultConfig, merge_framework_config
from .modules.action_dit import ActionDiT
from .modules.causal_norm import CausalRunningNormalization
from .modules.encoder import DINOEncoder
from .modules.feature_calibration import FeatureCalibration
from .modules.flow_matching import ContinuousFlowMatchScheduler
from .modules.mot import (
    BranchKVCache,
    InteractionRule,
    MoT,
    MoTBranchBinding,
    MoTExpertState,
    compile_attention_mask,
)
from .modules.reconstruction_probe import ReconstructionProbe
from .modules.temporal_bottleneck import TemporalRepresentationBottleneck
from .modules.world_dit import WorldDiT

REWAM_INTERACTION = [
    InteractionRule("state.video", "state.video", "per_frame_causal"),
    InteractionRule("action.action", "state.video", "key_first_frame"),
    InteractionRule("action.action", "action.action", "full"),
]

REWAM_JOINT_INTERACTION = [
    InteractionRule("state.video", "state.video", "per_frame_causal"),
    InteractionRule("action.action", "state.video", "full"),
    InteractionRule("action.action", "action.action", "full"),
]


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb_video(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert ``[T,H,W,3]`` RGB frames to ``[3,T,H,W]`` in ``[-1,1]``."""

    array = _as_numpy(value)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(f"RGB view must be [T,H,W,3] or [H,W,3], got {array.shape}.")
    if array.shape[-1] == 3:
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(3, 0, 1, 2)
    elif array.shape[1] == 3:
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(1, 0, 2, 3)
    else:
        raise ValueError(f"RGB view must have three channels, got {array.shape}.")
    tensor = tensor.to(device=device, dtype=torch.float32)
    if np.issubdtype(array.dtype, np.integer) or float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor.mul(2.0).sub(1.0).to(dtype=dtype)


def _resize_video(video: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    _, _, height, width = video.shape
    if (height, width) == size:
        return video
    images = video.permute(1, 0, 2, 3)
    images = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
    return images.permute(1, 0, 2, 3).contiguous()


class ReWAM(nn.Module):
    """The final, full ReWAM representation/action model."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        from omegaconf import OmegaConf

        if config is None or isinstance(config, dict):
            config = OmegaConf.create(config or {})
        self.config = merge_framework_config(ReWAMDefaultConfig, config)
        fw = self.config.framework
        if fw.name == "ReWAMJoint":
            if fw.variant != "base" and fw.variant != "joint":
                raise ValueError("ReWAMJoint requires joint variant.")
            fw.variant = "joint"
        self.variant = str(fw.variant)
        if self.variant not in ("base", "joint"):
            raise ValueError(f"ReWAM variant must be 'base' or 'joint', got {self.variant!r}.")
        self.interaction_rules = REWAM_JOINT_INTERACTION if self.variant == "joint" else REWAM_INTERACTION

        self.vision_encoder = DINOEncoder(**dict(fw.vision_encoder))
        self.feature_calibration = FeatureCalibration(**dict(fw.feature_calibration))
        tokenizer_cfg = fw.temporal_bottleneck
        self.state_dim = int(tokenizer_cfg.output_dim)
        if tokenizer_cfg.gradient_mode != "action_only":
            raise ValueError("The released ReWAM recipe requires action_only AGRS.")
        self.trb = TemporalRepresentationBottleneck(
            input_dim=self.vision_encoder.output_dim,
            output_dim=self.state_dim,
            patch_size=tokenizer_cfg.patch_size,
            qk_dim=int(tokenizer_cfg.qk_dim),
            num_heads=int(tokenizer_cfg.num_heads),
            norm_eps=float(tokenizer_cfg.norm_eps),
            rope_base=float(tokenizer_cfg.rope_base),
            position_scale=float(tokenizer_cfg.position_scale),
        )
        normalization = tokenizer_cfg.causal_normalization
        self.causal_norm = CausalRunningNormalization(
            self.state_dim,
            eps=float(normalization.eps),
            momentum=float(normalization.momentum),
        )
        probe_cfg = fw.reconstruction_probe
        self.reconstruction_probe = (
            ReconstructionProbe(
                self.state_dim,
                self.vision_encoder.output_dim,
                patch_size=self.trb.patch_size,
                decoder_path=probe_cfg.decoder_path,
            )
            if probe_cfg is not None
            else None
        )
        self.reconstruction_probe_loss_weight = float(probe_cfg.loss_weight) if probe_cfg is not None else 0.0
        if self.reconstruction_probe_loss_weight < 0:
            raise ValueError("framework.reconstruction_probe.loss_weight must be non-negative.")

        state_cfg = fw.world_dit_config
        state_patch = tuple(int(value) for value in state_cfg.patch_size)
        if state_patch != (1, 1, 1):
            raise ValueError(f"ReWAM world_dit_config.patch_size must be [1,1,1], got {state_patch}.")
        self.world_expert = WorldDiT(
            hidden_dim=int(state_cfg.hidden_dim),
            in_dim=self.state_dim,
            ffn_dim=int(state_cfg.ffn_dim),
            out_dim=self.state_dim,
            text_dim=int(state_cfg.text_dim),
            freq_dim=int(state_cfg.freq_dim),
            eps=float(state_cfg.eps),
            patch_size=state_patch,
            num_heads=int(state_cfg.num_heads),
            attn_head_dim=int(state_cfg.attn_head_dim),
            num_layers=int(state_cfg.num_layers),
        )
        action_cfg = fw.action_dit_config
        self.action_dim = int(action_cfg.action_dim)
        self.action_expert = ActionDiT(
            hidden_dim=int(action_cfg.hidden_dim),
            action_dim=self.action_dim,
            ffn_dim=int(action_cfg.ffn_dim),
            text_dim=int(action_cfg.text_dim),
            freq_dim=int(action_cfg.freq_dim),
            eps=float(action_cfg.eps),
            num_heads=int(action_cfg.num_heads),
            attn_head_dim=int(action_cfg.attn_head_dim),
            num_layers=int(action_cfg.num_layers),
        )
        self.mot = MoT(gradient_checkpoint=bool(fw.mot_checkpoint_mixed_attn))
        problems = self.mot.validate_expert(self.world_expert) + self.mot.validate_expert(self.action_expert)
        if problems:
            raise ValueError(f"ReWAM MoT expert validation failed: {problems}")

        self.num_history_frames = self.trb.patch_size[0] - 1
        world_scheduler = fw.world_scheduler
        self._world_train_shift = world_scheduler.train_shift
        self._world_infer_shift = world_scheduler.infer_shift
        self._world_shift_base_dim = int(world_scheduler.shift_base_dim)
        self._world_shift_scope = str(world_scheduler.shift_scope)
        if self._world_shift_base_dim <= 0 or self._world_shift_scope not in {"future_tokens", "all_tokens"}:
            raise ValueError("Invalid world_scheduler shift_base_dim or shift_scope.")
        self.world_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=int(world_scheduler.num_train_timesteps),
            shift=1.0 if self._world_train_shift == "auto" else float(self._world_train_shift),
        )
        action_scheduler = fw.action_scheduler
        self.action_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=int(action_scheduler.num_train_timesteps),
            shift=float(action_scheduler.train_shift),
        )
        self.action_infer_shift = float(action_scheduler.infer_shift)
        self.action_num_inference_steps = int(action_scheduler.num_inference_steps)
        self.compile_action_infer = bool(fw.compile_action_infer)
        self.world_loss_weight = float(fw.loss.lambda_world)
        self.action_loss_weight = float(fw.loss.lambda_action)
        self.proprio_dim = None if fw.proprio_dim is None else int(fw.proprio_dim)
        self.proprio_encoder = (
            nn.Linear(self.proprio_dim, int(state_cfg.text_dim)) if self.proprio_dim is not None else None
        )
        self.prompt_template = str(fw.text_encoder.prompt_template)
        if "{lang}" not in self.prompt_template:
            raise ValueError("framework.text_encoder.prompt_template must contain {lang}.")
        self.text_encoder = None
        self.tokenizer = None
        if bool(fw.load_text_encoder):
            self.attach_text_encoder(fw.text_encoder.pretrained_path)
        self._inference_context_cache = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._inference_context_cache = None
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def attach_text_encoder(self, pretrained_path: str) -> None:
        if not pretrained_path:
            raise ValueError("ReWAM text_encoder.pretrained_path is required for online text encoding.")
        from .modules.text_encoder import TransformersT5Encoder, TransformersTokenizer

        text_cfg = self.config.framework.text_encoder
        self.text_encoder = TransformersT5Encoder(
            pretrained_model_name_or_path=pretrained_path,
            subfolder="text_encoder",
            torch_dtype=next(self.parameters()).dtype,
            device="cpu",
        )
        self.tokenizer = TransformersTokenizer(
            pretrained_model_name_or_path=pretrained_path,
            subfolder="tokenizer",
            seq_len=int(text_cfg.max_length),
        )

    def _encode_context(self, prompts: list[str], device: torch.device, dtype: torch.dtype):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("ReWAM needs cached context/context_mask or an attached text encoder.")
        formatted = [self.prompt_template.format(lang=prompt) for prompt in prompts]
        cache_key = (tuple(formatted), device, dtype)
        if not self.training and self._inference_context_cache is not None:
            key, context, mask = self._inference_context_cache
            if key == cache_key:
                return context, mask
        ids, mask = self.tokenizer(formatted, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device=device, dtype=torch.bool)
        context = self.text_encoder(ids, mask).to(dtype=dtype)
        context = context.masked_fill(~mask.unsqueeze(-1), 0)
        mask = torch.ones_like(mask)
        if not self.training:
            self._inference_context_cache = (cache_key, context, mask)
        return context, mask

    def _resolve_context(self, examples: list[dict], device: torch.device, dtype: torch.dtype):
        presence = [("context" in example, "context_mask" in example) for example in examples]
        if any(has_context != has_mask for has_context, has_mask in presence):
            raise ValueError("Each ReWAM example must provide both context and context_mask, or neither.")
        if all(has_context for has_context, _ in presence):
            context = torch.stack([torch.as_tensor(example["context"]) for example in examples])
            mask = torch.stack(
                [torch.as_tensor(example["context_mask"], dtype=torch.bool) for example in examples]
            )
            context = context.to(device=device, dtype=dtype, non_blocking=True)
            mask = mask.to(device=device, dtype=torch.bool, non_blocking=True)
        elif any(has_context for has_context, _ in presence):
            raise ValueError("Cached text context must be present for every ReWAM batch item.")
        else:
            context, mask = self._encode_context(
                [str(example["lang"]) for example in examples], device, dtype
            )
        if self.proprio_encoder is not None:
            states = []
            for example in examples:
                state = _as_numpy(example["state"])
                if state.ndim == 2:
                    state = state[-1]
                if state.shape != (self.proprio_dim,):
                    raise ValueError(f"ReWAM state must be [{self.proprio_dim}], got {state.shape}.")
                states.append(torch.from_numpy(np.asarray(state, dtype=np.float32)))
            proprio = torch.stack(states).to(device=device, dtype=dtype)
            token = self.proprio_encoder(proprio).unsqueeze(1).to(dtype)
            context = torch.cat((context, token), dim=1)
            mask = torch.cat(
                (mask, torch.ones((len(examples), 1), device=device, dtype=torch.bool)),
                dim=1,
            )
        return context, mask

    def _prepare_views(
        self,
        example: dict,
        *,
        training: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if "rewam_video" in example:
            source_views = example["rewam_video"]
        else:
            current = list(example["image"])
            if len(current) != 3:
                raise ValueError(f"ReWAM requires three camera images, got {len(current)}.")
            if "history_image" in example:
                history = [_as_numpy(view) for view in example["history_image"]]
            elif "previous_image" in example:
                # Existing host datasets expose one historical image per view.
                if self.num_history_frames != 1:
                    raise ValueError("Multiple TRB history frames require history_image [3,T_history,H,W,3].")
                history = [_as_numpy(image)[None] for image in example["previous_image"]]
            else:
                history = [
                    np.repeat(_as_numpy(image)[None], self.num_history_frames, axis=0) for image in current
                ]
            if len(history) != 3 or any(view.shape[0] != self.num_history_frames for view in history):
                raise ValueError(
                    "history_image must contain the configured history frames for all three cameras."
                )
            future = _as_numpy(example["future_imgs"]) if training else None
            if training and (future.ndim != 5 or future.shape[0] != 3):
                raise ValueError(
                    f"ReWAM future_imgs must be [3,T,H,W,3], got {getattr(future, 'shape', None)}."
                )
            source_views = []
            for view in range(3):
                frames = [*history[view], _as_numpy(current[view])]
                if future is not None:
                    frames.extend(future[view])
                source_views.append(np.stack(frames))

        if len(source_views) != 3:
            raise ValueError(f"ReWAM requires three camera videos, got {len(source_views)}.")
        patch_t = self.trb.patch_size[0]
        sizes = ((256, 320), (128, 160), (128, 160))
        source_frames = None
        views = []
        for source, size in zip(source_views, sizes, strict=True):
            video = _rgb_video(source, device=device, dtype=dtype)
            # The trainer's periodic action sanity check calls predict_action
            # with an unmodified training example.  Such examples contain the
            # full [-stride, 0, stride, ..., horizon] clip, while inference
            # consumes only the first group (history and current observations).
            if not training:
                video = video[:, :patch_t]
            frames = video.shape[1]
            if frames % patch_t or frames < (2 * patch_t if training else patch_t):
                raise ValueError(
                    f"ReWAM video length {frames} must contain complete TRB groups of {patch_t}."
                )
            if source_frames is not None and frames != source_frames:
                raise ValueError("All ReWAM camera views must have the same frame count.")
            source_frames = frames
            views.append(_resize_video(video, size))

        if training:
            source_pad = example.get("rewam_video_is_pad")
            if source_pad is None:
                future_pad = np.asarray(
                    example.get("future_is_pad", np.zeros(source_frames - patch_t)),
                    dtype=bool,
                )
                history_pad = np.asarray(
                    example.get(
                        "history_is_pad",
                        [bool(example.get("previous_is_pad", False))] * self.num_history_frames,
                    ),
                    dtype=bool,
                )
                if history_pad.shape != (self.num_history_frames,):
                    raise ValueError("history_is_pad must match the number of historical frames.")
                source_pad = np.concatenate(
                    (
                        np.concatenate((history_pad, [False])),
                        future_pad,
                    )
                )
            source_pad = np.asarray(source_pad, dtype=bool)
            if source_pad.shape != (source_frames,):
                raise ValueError(f"rewam_video_is_pad must be [{source_frames}], got {source_pad.shape}.")
            state_pad = (
                torch.from_numpy(source_pad).to(device).view(source_frames // patch_t, patch_t).all(dim=1)
            )
        else:
            state_pad = torch.zeros(1, device=device, dtype=torch.bool)
        return views, state_pad

    def _encode_view_features(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        batch = views[0].shape[0]
        front = self.feature_calibration(
            self.vision_encoder(views[0], layers=self.feature_calibration.layers)
        )
        wrists = self.feature_calibration(
            self.vision_encoder(
                torch.cat((views[1], views[2]), dim=0), layers=self.feature_calibration.layers
            )
        )
        left, right = wrists.split(batch, dim=0)
        return [front, left, right]

    @staticmethod
    def _merge_view_states(view_states: list[torch.Tensor]) -> torch.Tensor:
        front, left, right = view_states
        return torch.cat((front, torch.cat((left, right), dim=-1)), dim=-2).contiguous()

    @staticmethod
    def _split_view_states(states: torch.Tensor) -> list[torch.Tensor]:
        height, width = states.shape[-2:]
        if height % 3 or width % 2:
            raise ValueError(f"Invalid Robotwin state layout {height}x{width}.")
        top = height * 2 // 3
        return [states[..., :top, :], states[..., top:, : width // 2], states[..., top:, width // 2 :]]

    def _encode_raw_states(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self._encode_view_features(views)
        view_states = self.trb(features)
        states = self._merge_view_states(view_states)
        return states, features

    def _normalize_state_latents(
        self,
        raw_states: torch.Tensor,
        *,
        update_stats: bool | None = None,
    ) -> torch.Tensor:
        return self.causal_norm(raw_states, update_stats=update_stats)

    @staticmethod
    def _token_layout(states: torch.Tensor, state_is_pad: torch.Tensor):
        batch, _, frames, height, width = states.shape
        if state_is_pad.shape != (batch, frames):
            raise ValueError(f"state_is_pad must be [B,T]={batch, frames}, got {tuple(state_is_pad.shape)}.")
        tokens_per_frame = height * width
        valid = (~state_is_pad).unsqueeze(-1).expand(-1, -1, tokens_per_frame).reshape(batch, -1)
        frame_ids = (
            torch.arange(frames, device=states.device)
            .view(1, frames, 1)
            .expand(batch, -1, tokens_per_frame)
            .reshape(batch, -1)
        )
        return valid, frame_ids

    def _reconstruction_loss(
        self, raw_states: torch.Tensor, feature_targets: list[torch.Tensor]
    ) -> torch.Tensor:
        if self.reconstruction_probe is None or self.reconstruction_probe_loss_weight == 0:
            return raw_states.new_zeros(())
        predictions = [self.reconstruction_probe(view) for view in self._split_view_states(raw_states)]
        return torch.stack(
            [
                F.mse_loss(prediction.float(), target.detach().float())
                for prediction, target in zip(predictions, feature_targets, strict=True)
            ]
        ).mean()

    def _resolve_world_shift(self, value: float | str, states: torch.Tensor) -> float:
        if value != "auto":
            return float(value)
        _, channels, frames, height, width = states.shape
        if self._world_shift_scope == "future_tokens":
            frames -= 1
        return math.sqrt(channels * frames * height * width / self._world_shift_base_dim)

    def _training_objective(
        self,
        *,
        raw_states: torch.Tensor,
        states: torch.Tensor,
        feature_targets: list[torch.Tensor],
        state_is_pad: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: torch.Tensor,
        action_dim_mask: torch.Tensor | None = None,
    ) -> dict:
        if states.shape[2] < 2:
            raise ValueError("ReWAM training requires current and future TRB groups.")
        shift = self._resolve_world_shift(self._world_train_shift, states)
        if self.world_scheduler.shift != shift:
            self.world_scheduler = ContinuousFlowMatchScheduler(
                num_train_timesteps=self.world_scheduler.num_train_timesteps, shift=shift
            )
        online_states = states
        online_current = online_states[:, :, :1]
        current_proxy = online_current.detach()
        if torch.is_grad_enabled():
            current_proxy.requires_grad_(True)
        model_states = torch.cat((current_proxy, online_states[:, :, 1:].detach()), dim=2)

        batch, _, frames, _, _ = model_states.shape
        future_t = self.world_scheduler.sample_training_t(
            batch * (frames - 1), model_states.device, model_states.dtype
        ).view(batch, frames - 1)
        state_t = torch.cat((torch.zeros_like(future_t[:, :1]), future_t), dim=1)
        state_noise = torch.randn_like(model_states)
        noisy_states = self.world_scheduler.add_noise(model_states, state_noise, state_t)
        noisy_states[:, :, :1] = model_states[:, :, :1]
        state_target = self.world_scheduler.training_target(model_states, state_noise, state_t)

        if action_dim_mask is not None:
            if action_dim_mask.shape != action.shape:
                raise ValueError("action_dim_mask must match the action shape.")
            # Missing human joints have a zero clean endpoint, including when
            # callers supply non-finite placeholders in inactive dimensions.
            action = torch.where(action_dim_mask, action, torch.zeros_like(action))
        action_t = self.action_scheduler.sample_training_t(batch, action.device, action.dtype)
        action_noise = torch.randn_like(action)
        noisy_action = self.action_scheduler.add_noise(action, action_noise, action_t)
        action_target = self.action_scheduler.training_target(action, action_noise, action_t)

        state_valid, frame_ids = self._token_layout(noisy_states, state_is_pad)
        state_state = self.world_expert.prepare(
            noisy_states, state_t, context, context_mask, state_valid, frame_ids
        )
        action_state = self.action_expert.prepare(noisy_action, action_t, context, context_mask)
        state_binding = MoTBranchBinding("state", self.world_expert, state_state)
        action_binding = MoTBranchBinding("action", self.action_expert, action_state)
        mask = compile_attention_mask(
            query_bindings=[state_binding, action_binding],
            key_bindings=[state_binding, action_binding],
            rules=self.interaction_rules,
        )
        output = self.mot.forward([state_binding, action_binding], mask)
        predicted_state = self.world_expert.mot_finalize(output["state"])[:, :, 1:]
        predicted_action = self.action_expert.mot_finalize(output["action"])

        state_error = F.mse_loss(
            predicted_state.float(), state_target[:, :, 1:].float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        state_valid_frames = (~state_is_pad[:, 1:]).to(state_error.dtype)
        world_weights = self.world_scheduler.training_weight(state_t[:, 1:]).to(state_error.dtype)
        state_loss = (
            (state_error * world_weights * state_valid_frames).sum(dim=1)
            / state_valid_frames.sum(dim=1).clamp_min(1.0)
        ).mean()

        action_error = F.mse_loss(predicted_action.float(), action_target.float(), reduction="none")
        action_valid = (~action_is_pad).unsqueeze(-1).expand_as(action_error)
        if action_dim_mask is not None:
            action_valid = action_valid & action_dim_mask
        action_valid = action_valid.to(action_error.dtype)
        action_loss_per_sample = (action_error * action_valid).sum(dim=(1, 2)) / action_valid.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
        action_weights = self.action_scheduler.training_weight(action_t).to(action_error.dtype)
        action_loss = (action_loss_per_sample * action_weights).mean()

        route_loss = action_loss.new_zeros(())
        proxy_gradient_rms = 0.0
        if torch.is_grad_enabled():
            proxy_gradient = torch.autograd.grad(
                self.action_loss_weight * action_loss,
                current_proxy,
                retain_graph=True,
                create_graph=False,
            )[0]
            route_loss = ((online_current - online_current.detach()) * proxy_gradient.detach()).sum()
            proxy_gradient_rms = float(proxy_gradient.detach().float().square().mean().sqrt().item())

        reconstruction_loss = self._reconstruction_loss(raw_states, feature_targets)
        total = (
            self.world_loss_weight * state_loss
            + self.action_loss_weight * action_loss
            + self.reconstruction_probe_loss_weight * reconstruction_loss
            + route_loss
        )
        entropy = self.trb.last_attention_entropy
        metrics = {
            "loss_world": float((self.world_loss_weight * state_loss).detach().item()),
            "loss_action": float((self.action_loss_weight * action_loss).detach().item()),
            "loss_reconstruction_probe": float(
                (self.reconstruction_probe_loss_weight * reconstruction_loss).detach().item()
            ),
            "trb_action_gradient_rms": proxy_gradient_rms,
            "trb_attention_entropy": (float(entropy.item()) if entropy is not None else 0.0),
            "state_raw_mean": float(raw_states.detach().float().mean().item()),
            "state_raw_std": float(raw_states.detach().float().std().item()),
        }
        return {"action_loss": total, "loss_metrics": metrics}

    def forward(self, examples: list[dict], **kwargs: Any) -> dict:
        del kwargs
        if not isinstance(examples, list) or not examples:
            raise ValueError("ReWAM forward expects a non-empty List[dict].")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        context, context_mask = self._resolve_context(examples, device, dtype)

        per_example_views = [
            self._prepare_views(example, training=True, device=device, dtype=dtype) for example in examples
        ]
        views = [torch.stack([item[0][view_index] for item in per_example_views]) for view_index in range(3)]
        state_is_pad = torch.stack([item[1] for item in per_example_views])
        raw_states, feature_targets = self._encode_raw_states(views)
        states = self._normalize_state_latents(raw_states, update_stats=self.training)

        action = torch.stack(
            [torch.as_tensor(example["action"], device=device, dtype=dtype) for example in examples]
        )
        if action.ndim != 3 or action.shape[0] != len(examples) or action.shape[2] != self.action_dim:
            raise ValueError(f"ReWAM action must be [B,H,{self.action_dim}], got {tuple(action.shape)}.")
        action_is_pad = torch.stack(
            [
                torch.as_tensor(
                    example.get("action_is_pad", np.zeros(action.shape[1])),
                    device=device,
                    dtype=torch.bool,
                )
                for example in examples
            ]
        )
        return self._training_objective(
            raw_states=raw_states,
            states=states,
            feature_targets=feature_targets,
            state_is_pad=state_is_pad,
            context=context,
            context_mask=context_mask,
            action=action,
            action_is_pad=action_is_pad,
            action_dim_mask=self._action_dimension_mask(examples, device, action.shape[1]),
        )

    def _action_dimension_mask(
        self, examples: list[dict], device: torch.device, action_horizon: int
    ) -> torch.Tensor | None:
        if not any("action_dim_mask" in example for example in examples):
            return None
        shape = (action_horizon, self.action_dim)
        masks = []
        for example in examples:
            mask = torch.as_tensor(
                example.get("action_dim_mask", np.ones(shape, dtype=bool)),
                device=device,
                dtype=torch.bool,
            )
            if tuple(mask.shape) != shape:
                raise ValueError(f"action_dim_mask must be {shape}, got {tuple(mask.shape)}.")
            masks.append(mask)
        return torch.stack(masks)

    def _pin_inactive_actions(self, actions, initial_noise, active, timestep, delta):
        if active is None:
            return actions
        sigma_next = timestep / float(self.action_scheduler.num_train_timesteps) + delta
        return torch.where(active, actions, initial_noise * sigma_next)

    def _base_inference_core(
        self, action: MoTExpertState, cache: BranchKVCache, mask: torch.Tensor
    ) -> torch.Tensor:
        binding = MoTBranchBinding("action", self.action_expert, action)
        output = self.mot.forward_with_cache([binding], [cache], mask)
        return self.action_expert.mot_finalize(output["action"])

    def _joint_inference_core(
        self, state: MoTExpertState, action: MoTExpertState, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Use the same MoT attention and expert protocol without training-only
        # payload identity/checkpoint guards, which cannot enter a full graph.
        state_len = state.tokens.shape[1]
        for layer in range(self.world_expert.mot_num_layers):
            state_io = self.world_expert.mot_layer_qkv(layer, state)
            action_io = self.action_expert.mot_layer_qkv(layer, action)
            mixed = self.mot._mixed_attention(
                torch.cat((state_io.q, action_io.q), dim=2),
                torch.cat((state_io.k, action_io.k), dim=2),
                torch.cat((state_io.v, action_io.v), dim=2),
                mask,
                self.world_expert.mot_num_heads,
            )
            state = self.world_expert.mot_layer_post_attention(
                layer, mixed[:, :, :state_len], state_io, state
            )
            action = self.action_expert.mot_layer_post_attention(
                layer, mixed[:, :, state_len:], action_io, action
            )
        return (
            self.world_expert.mot_finalize(state)[:, :, 1:],
            self.action_expert.mot_finalize(action),
        )

    def _get_inference_core(self, compiled: bool) -> Callable:
        core = self._joint_inference_core if self.variant == "joint" else self._base_inference_core
        if not compiled:
            return core
        if not hasattr(self, "_compiled_inference_core"):
            self._compiled_inference_core = torch.compile(core, mode="reduce-overhead", fullgraph=True)
        return self._compiled_inference_core

    @torch.inference_mode()
    def predict_action(
        self,
        examples: list[dict],
        *,
        action_horizon: int,
        num_future_frames: int | None = None,
        **kwargs: Any,
    ) -> dict:
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if self.variant == "joint":
            if (
                num_future_frames is None
                or num_future_frames <= 0
                or num_future_frames % self.trb.patch_size[0]
            ):
                raise ValueError("Joint inference requires num_future_frames divisible by TRB patch_size[0].")
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        requested_steps = kwargs.pop("num_ddim_steps", None)
        if num_inference_steps is not None and requested_steps is not None:
            raise ValueError("Set only num_inference_steps; num_ddim_steps also specifies the step count.")
        if num_inference_steps is None:
            num_inference_steps = requested_steps or self.action_num_inference_steps
        sampler_seed = kwargs.pop("sampler_seed", None)
        compiled = kwargs.pop("compile_action_infer", self.compile_action_infer)
        if not isinstance(compiled, bool):
            raise TypeError("compile_action_infer must be a bool.")
        kwargs.pop("use_ddim", None)
        kwargs.pop("do_sample", None)
        if kwargs:
            raise ValueError(f"Unknown ReWAM predict_action kwargs: {sorted(kwargs)}")
        if not isinstance(examples, list):
            examples = [examples]
        if not examples:
            raise ValueError("ReWAM predict_action expects a non-empty List[dict].")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        context, context_mask = self._resolve_context(examples, device, dtype)
        per_example_views = [
            self._prepare_views(example, training=False, device=device, dtype=dtype) for example in examples
        ]
        views = [torch.stack([item[0][view_index] for item in per_example_views]) for view_index in range(3)]
        raw_state, _ = self._encode_raw_states(views)
        current_state = self._normalize_state_latents(raw_state, update_stats=False)
        active = self._action_dimension_mask(examples, device, action_horizon)

        if self.variant == "joint":
            actions = self._predict_action_joint(
                current_state,
                context,
                context_mask,
                int(num_inference_steps),
                sampler_seed,
                active,
                action_horizon=action_horizon,
                num_future_frames=int(num_future_frames),
                compiled=compiled,
            )
            return {"normalized_actions": actions.detach().cpu().float().numpy()}

        batch = len(examples)
        state_pad = torch.zeros((batch, 1), device=device, dtype=torch.bool)
        state_valid, frame_ids = self._token_layout(current_state, state_pad)
        state_t = torch.zeros((batch, 1), device=device, dtype=dtype)
        state = self.world_expert.prepare(
            current_state, state_t, context, context_mask, state_valid, frame_ids
        )
        state_binding = MoTBranchBinding("state", self.world_expert, state)
        state_mask = compile_attention_mask(
            query_bindings=[state_binding],
            key_bindings=[state_binding],
            rules=self.interaction_rules,
        )
        cache = self.mot.prefill_branch_cache(state_binding, state_mask, export_groups=("video",))

        generator = None
        if sampler_seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(sampler_seed))
        actions = torch.randn(
            batch,
            action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        timesteps, deltas = self.action_scheduler.build_inference_schedule(
            int(num_inference_steps),
            device,
            dtype,
            shift_override=self.action_infer_shift,
        )
        initial_noise = actions / (timesteps[0] / float(self.action_scheduler.num_train_timesteps))
        first_t = timesteps[0].reshape(1).expand(batch)
        action_state = self.action_expert.prepare(actions, first_t, context, context_mask)
        action_binding = MoTBranchBinding("action", self.action_expert, action_state)
        mask = compile_attention_mask(
            query_bindings=[action_binding],
            key_bindings=[cache, action_binding],
            rules=self.interaction_rules,
        )
        conditioning = action_state
        denoise = self._get_inference_core(compiled)
        for index, (timestep, delta) in enumerate(zip(timesteps, deltas, strict=True)):
            if index:
                action_t = timestep.reshape(1).expand(batch)
                action_state = self.action_expert.prepare_step(actions, action_t, conditioning)
            if compiled:
                torch.compiler.cudagraph_mark_step_begin()
            prediction = denoise(action_state, cache, mask)
            actions = self.action_scheduler.step(prediction, delta, actions)
            actions = self._pin_inactive_actions(actions, initial_noise, active, timestep, delta)
            if compiled:
                # Detach live scheduler state from CUDA Graph-owned output storage.
                actions = actions.clone()
        return {"normalized_actions": actions.detach().cpu().float().numpy()}

    def _predict_action_joint(
        self,
        current_state: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        num_inference_steps: int,
        sampler_seed: int | None,
        active: torch.Tensor | None = None,
        *,
        compiled: bool = False,
        action_horizon: int,
        num_future_frames: int,
    ) -> torch.Tensor:
        """Denoise future states and actions together, keeping the current temporal group fixed."""
        batch, channels, _, height, width = current_state.shape
        state_steps = num_future_frames // self.trb.patch_size[0] + 1
        device, dtype = current_state.device, current_state.dtype
        generator = None
        if sampler_seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(sampler_seed))
        # Draw actions first so a fixed seed starts Base and Joint with the
        # same action noise. Future states are independent subsequent draws.
        actions = torch.randn(
            batch,
            action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        future = torch.randn(
            batch,
            channels,
            state_steps - 1,
            height,
            width,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        state_times, state_deltas = self.world_scheduler.build_inference_schedule(
            num_inference_steps,
            device,
            dtype,
            shift_override=self._resolve_world_shift(
                self._world_infer_shift, torch.cat((current_state, future), dim=2)
            ),
        )
        action_times, action_deltas = self.action_scheduler.build_inference_schedule(
            num_inference_steps, device, dtype, shift_override=self.action_infer_shift
        )
        initial_noise = actions / (action_times[0] / float(self.action_scheduler.num_train_timesteps))
        state_pad = torch.zeros((batch, state_steps), device=device, dtype=torch.bool)
        state_valid, frame_ids = self._token_layout(torch.cat((current_state, future), dim=2), state_pad)
        mask = None
        state_conditioning = action_conditioning = None
        denoise = self._get_inference_core(compiled)
        for state_time, state_delta, action_time, action_delta in zip(
            state_times, state_deltas, action_times, action_deltas, strict=True
        ):
            states = torch.cat((current_state, future), dim=2)
            state_t = state_time.expand(batch, state_steps).clone()
            state_t[:, 0] = 0
            if state_conditioning is None:
                state = self.world_expert.prepare(
                    states, state_t, context, context_mask, state_valid, frame_ids
                )
                action = self.action_expert.prepare(actions, action_time.expand(batch), context, context_mask)
                state_conditioning, action_conditioning = state, action
            else:
                state = self.world_expert.prepare_step(states, state_t, state_conditioning)
                action = self.action_expert.prepare_step(
                    actions, action_time.expand(batch), action_conditioning
                )
            if mask is None:
                bindings = [
                    MoTBranchBinding("state", self.world_expert, state),
                    MoTBranchBinding("action", self.action_expert, action),
                ]
                mask = compile_attention_mask(
                    query_bindings=bindings,
                    key_bindings=bindings,
                    rules=self.interaction_rules,
                )
            if compiled:
                torch.compiler.cudagraph_mark_step_begin()
            state_velocity, action_velocity = denoise(state, action, mask)
            future = self.world_scheduler.step(state_velocity, state_delta, future)
            actions = self.action_scheduler.step(action_velocity, action_delta, actions)
            actions = self._pin_inactive_actions(actions, initial_noise, active, action_time, action_delta)
            if compiled:
                future, actions = future.clone(), actions.clone()
        return actions


__all__ = [
    "REWAM_INTERACTION",
    "REWAM_JOINT_INTERACTION",
    "ReWAM",
    "ReWAMDefaultConfig",
]
