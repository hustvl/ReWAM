import json
import logging
import os
import re
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from .utils.random import set_global_seed, worker_init_function
from .utils.samplers import ResumableEpochSampler
from .utils.video import pil_frames_to_video_tensor, save_mp4, video_psnr, video_ssim

logger = logging.getLogger(__name__)


class ReWAMTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)

        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        tensorboard_cfg = cfg.get("tensorboard", None)
        self.tensorboard_enabled = bool(tensorboard_cfg is not None and tensorboard_cfg.get("enabled", False))

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get(
                "stage", "unknown"
            ),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("Training device: %s", self.accelerator.device)
        set_global_seed(self.seed)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.mot.parameters())
        proprio_encoder = self.model.proprio_encoder
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        reconstruction_probe = self.model.reconstruction_probe
        if reconstruction_probe is not None:
            trainable_params.extend(list(reconstruction_probe.parameters()))
        trb = self.model.trb
        trainable_params.extend(list(trb.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_function)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_root).mkdir(parents=True, exist_ok=True)
        Path(self.weights_dir).mkdir(parents=True, exist_ok=True)
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        Path(self.eval_dir).mkdir(parents=True, exist_ok=True)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self.tensorboard_writer = None
        self._init_wandb()
        self._init_tensorboard()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=(None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group)),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _init_tensorboard(self):
        if not self.tensorboard_enabled or not self.accelerator.is_main_process:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "TensorBoard logging is enabled in config (`tensorboard.enabled=true`) "
                "but tensorboard is not installed."
            ) from e

        tb_cfg = self.cfg.tensorboard
        log_dir = tb_cfg.get("log_dir", None)
        if log_dir in (None, "null", ""):
            log_dir = os.path.join(self.output_dir, "tensorboard")
        self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
        logger.info("Initialized tensorboard writer: log_dir=%s", log_dir)

    def _tensorboard_log(self, payload: dict):
        if self.tensorboard_writer is None:
            return
        for key, value in payload.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.tensorboard_writer.add_scalar(key, float(value), self.global_step)
        self.tensorboard_writer.flush()

    def _tensorboard_log_video(self, tag: str, video: torch.Tensor, fps: int = 8):
        if self.tensorboard_writer is None:
            return
        video = video.detach().float().cpu().clamp(0.0, 1.0)
        video = video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        self.tensorboard_writer.add_video(tag, video, self.global_step, fps=fps)
        self.tensorboard_writer.flush()

    def _tensorboard_log_image(self, tag: str, image: torch.Tensor):
        if self.tensorboard_writer is None:
            return
        image = image.detach().float().cpu().clamp(0.0, 1.0)
        self.tensorboard_writer.add_image(tag, image, self.global_step, dataformats="CHW")
        self.tensorboard_writer.flush()

    def _finish_tensorboard(self):
        if self.tensorboard_writer is None:
            return
        self.tensorboard_writer.close()
        self.tensorboard_writer = None

    def _log_scalars(self, payload: dict):
        self._wandb_log(payload)
        self._tensorboard_log(payload)

    def _close_loggers(self):
        self._finish_wandb()
        self._finish_tensorboard()

    def _should_capture_representation_visualization(self) -> bool:
        return (
            self.tensorboard_writer is not None
            and self.accelerator.is_main_process
            and self.log_every > 0
            and self.accelerator.sync_gradients
            and (self.global_step + 1) % self.log_every == 0
        )

    @staticmethod
    def _set_representation_visualization_capture(model, enabled: bool):
        setter = getattr(model, "set_representation_visualization_capture", None)
        if callable(setter):
            setter(enabled)

    def _log_representation_visualization(self, model):
        pop_visualization = getattr(model, "pop_last_representation_visualization", None)
        if not callable(pop_visualization):
            return
        visualization = pop_visualization()
        if not isinstance(visualization, dict):
            return
        tag = visualization.get("tag")
        image = visualization.get("image")
        if isinstance(tag, str) and isinstance(image, torch.Tensor):
            self._tensorboard_log_image(tag, image)

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            logger.error(
                "Dataset lengths differ across ranks: dataset=%s lengths=%s",
                dataset_name,
                gathered_lengths.cpu().tolist(),
            )
        self.accelerator.wait_for_everyone()
        raise RuntimeError(f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}")

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded model weights; optimizer, scheduler and training progress were not restored.")

    def _set_dit_only_train_mode(self):
        logger.info("Enabled ReWAM trainable components; frozen encoders remain in evaluation mode")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.mot.train()
        model.mot.requires_grad_(True)
        proprio_encoder = model.proprio_encoder
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        reconstruction_probe = model.reconstruction_probe
        if reconstruction_probe is not None:
            reconstruction_probe.train()
            reconstruction_probe.requires_grad_(True)
        trb = model.trb
        trb.train()
        trb.requires_grad_(True)
        # Keep frozen encoders deterministic while updating the TRB's lagged statistics.
        causal_norm = model.causal_norm
        causal_norm.train()

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(
                f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}"
            )

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(
                f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}"
            )

        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(f"`sample['action']` must be a torch.Tensor, got {type(action)}")
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(
                    f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
                )
            # Do not infer action/world alignment from the raw source-video
            # length here. History-aware datasets prepend a conditioning frame,
            # and representation models may group/downsample source frames.
            # The concrete model owns validation against its effective temporal
            # representation in training_loss()/inference.
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    def _eval_video_metrics(
        self,
        model,
        video0: torch.Tensor,
        pred_video,
        decoded_gt_video=None,
        metric_gt_video=None,
    ):
        metrics = {}
        artifacts = {}
        if metric_gt_video is None:
            gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
        else:
            gt_video_tensor = pil_frames_to_video_tensor(metric_gt_video)
        if pred_video is None:
            return metrics, artifacts

        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )
        metrics["psnr_rg"] = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        metrics["ssim_rg"] = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)
        if decoded_gt_video is not None:
            decoded_gt_tensor = pil_frames_to_video_tensor(decoded_gt_video)
            assert decoded_gt_tensor.shape == gt_video_tensor.shape, (
                "Eval TRB reconstruction/GT shape mismatch: "
                f"decoded={tuple(decoded_gt_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
            )
            metrics["psnr_dg"] = video_psnr(pred=decoded_gt_tensor, target=gt_video_tensor)
            metrics["ssim_dg"] = video_ssim(pred=decoded_gt_tensor, target=gt_video_tensor)

        video_columns = [pred_video_tensor]
        if decoded_gt_video is not None:
            video_columns.append(decoded_gt_tensor)

        video_columns.append(gt_video_tensor)

        stitched_video_tensor = torch.cat(
            video_columns,
            dim=2,
        ).contiguous()
        artifacts["video_tensor"] = stitched_video_tensor
        if self.accelerator.is_main_process:
            stitched_frames = []
            for t in range(stitched_video_tensor.shape[1]):
                frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(
                    np.uint8
                )
                stitched_frames.append(Image.fromarray(frame))

            video_path = os.path.join(
                self.eval_dir,
                f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
            )
            save_mp4(stitched_frames, video_path, fps=8)
            artifacts["video_path"] = video_path
        return metrics, artifacts

    def _eval_action_metrics(
        self,
        *,
        sample: dict,
        action: torch.Tensor | None,
        pred_action,
    ) -> dict[str, float]:
        if action is None or pred_action is None:
            return {}
        if sample["proprio"] is None:
            raise ValueError("Eval sample must contain `proprio` for action denormalization.")

        processor = self.val_dataset.processor

        def denormalize(action_name: str, raw_action) -> torch.Tensor:
            if not isinstance(raw_action, torch.Tensor):
                raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
            if raw_action.ndim == 2:
                action_btd = raw_action.unsqueeze(0)
            elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                action_btd = raw_action
            else:
                raise ValueError(
                    f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                )
            denorm_action = processor.normalizer.normalizers["action"].backward(
                action_btd.detach().to(device="cpu", dtype=torch.float32)
            )
            if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                raise ValueError(
                    f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                )
            return denorm_action

        pred_action_denorm = denormalize("pred", pred_action)
        gt_action_denorm = denormalize("gt", action)
        if pred_action_denorm.shape != gt_action_denorm.shape:
            raise ValueError(
                "Predicted action/GT action shape mismatch after denormalization: "
                f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
            )
        action_diff = pred_action_denorm - gt_action_denorm
        return {
            "action_l1": action_diff.abs().mean().item(),
            "action_l2": action_diff.pow(2).mean().item(),
        }

    def _gather_optional_eval_metrics(self, val_loss: float, metrics: dict[str, float]) -> dict[str, float]:
        metric_names = [
            "psnr_rg",
            "ssim_rg",
            "psnr_rd",
            "ssim_rd",
            "psnr_dg",
            "ssim_dg",
            "action_l2",
            "action_l1",
        ]
        values = [float(val_loss)]
        for name in metric_names:
            present = name in metrics
            values.extend([1.0 if present else 0.0, float(metrics[name]) if present else 0.0])

        gathered = self.accelerator.gather_for_metrics(
            torch.tensor(values, device=self.accelerator.device, dtype=torch.float32).unsqueeze(0)
        )
        result = {"val_loss": float(gathered[:, 0].mean().item())}
        col = 1
        for name in metric_names:
            mask = gathered[:, col] > 0.5
            value_col = col + 1
            if bool(mask.any().item()):
                result[name] = float(gathered[mask, value_col].mean().item())
            col += 2
        return result

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_mot_training = model.mot.training
        model.eval()

        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0]  # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = (
            sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None
        )  # Use the current proprioceptive vector from [1,T,D].
        history_frames = model.num_history_frames
        input_image = video0[:, history_frames].unsqueeze(0)
        history_images = video0[:, :history_frames].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample["action_horizon"],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        infer_kwargs["history_images"] = history_images
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        decoded_gt_video = None
        metric_gt_video = None
        probe = model.reconstruction_probe
        if probe is not None and probe.decoder_path:
            joint_kwargs = dict(infer_kwargs)
            joint_kwargs["num_video_frames"] = joint_kwargs.pop("num_frames")
            joint_kwargs.pop("action_cfg_scale", None)
            pred = model.infer_joint(**joint_kwargs)
            decoded_gt_video = model.reconstruct_representation_video(video0.unsqueeze(0))
            metric_gt_video = model.prepare_representation_rgb_target(video0.unsqueeze(0))
        else:
            pred = model.infer(**infer_kwargs)

        pred_video = pred.get("video", None)
        pred_action = pred.get("action", None)

        local_metrics, artifacts = self._eval_video_metrics(
            model,
            video0,
            pred_video,
            decoded_gt_video=decoded_gt_video,
            metric_gt_video=metric_gt_video,
        )
        if metric_gt_video is not None:
            probe.release_decoder()
        local_metrics.update(
            self._eval_action_metrics(
                sample=sample,
                action=action,
                pred_action=pred_action,
            )
        )
        result = self._gather_optional_eval_metrics(val_loss, local_metrics)

        if was_mot_training:
            self._set_dit_only_train_mode()

        result.update(artifacts)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        Path(state_path).mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info(
            "Loaded accelerate training state from %s at step=%d",
            state_dir,
            self.global_step,
        )
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = (
                    self.model
                    if hasattr(self.model, "training_loss")
                    else self.accelerator.unwrap_model(self.model)
                )
                self._set_representation_visualization_capture(
                    unwrapped_model,
                    self._should_capture_representation_visualization(),
                )

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                trb = unwrapped_model.trb
                trb_grad_sq = None
                for parameter in trb.parameters():
                    if parameter.grad is None:
                        continue
                    value = parameter.grad.detach().float().pow(2).sum()
                    trb_grad_sq = value if trb_grad_sq is None else trb_grad_sq + value
                if trb_grad_sq is not None:
                    loss_dict["repr/trb_parameter_grad_norm"] = float(trb_grad_sq.sqrt().item())

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(
                            float(value), device=loss.device, dtype=torch.float32
                        ).reshape(1)
                        global_loss_metrics[key] = float(self.accelerator.gather(metric_tensor).mean().item())
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if (
                        self.log_every > 0
                        and self.global_step % self.log_every == 0
                        and self.accelerator.is_main_process
                    ):
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "Train: epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join(
                                [f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())]
                            )
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._log_scalars(wandb_payload)
                        self._log_representation_visualization(unwrapped_model)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "Evaluation: step=%d val_loss=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                            )
                            if "psnr_rd" in metrics:
                                description += " infer_psnr=%.4f" % metrics["psnr_rd"]
                            if "ssim_rd" in metrics:
                                description += " infer_ssim=%.4f" % metrics["ssim_rd"]
                            if "psnr_rg" in metrics:
                                description += " rollout_psnr=%.4f" % metrics["psnr_rg"]
                            if "psnr_dg" in metrics:
                                description += " decoder_psnr=%.4f" % metrics["psnr_dg"]
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                            }
                            for key in (
                                "psnr_rg",
                                "ssim_rg",
                                "psnr_rd",
                                "ssim_rd",
                                "psnr_dg",
                                "ssim_dg",
                                "action_l2",
                                "action_l1",
                            ):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            self._log_scalars(eval_payload)
                            if "video_tensor" in metrics:
                                probe = unwrapped_model.reconstruction_probe
                                if probe is not None and probe.decoder_path:
                                    video_tag = "eval/rollout_decoder_gt"
                                else:
                                    video_tag = (
                                        "eval/pred_reconstruction_gt"
                                        if "psnr_dg" in metrics
                                        else "eval/pred_gt"
                                    )
                                self._tensorboard_log_video(video_tag, metrics["video_tensor"], fps=8)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "Saved checkpoint: step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "Training reached max_steps: step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                            self._close_loggers()
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "Training finished: step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
            self._close_loggers()
