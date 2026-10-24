import logging
import os
from pathlib import Path

import torch
from accelerate.utils import set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .trainer import ReWAMTrainer
from .utils import work_dir
from .utils.config import as_plain_dict
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise TypeError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. " "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _validate_action_scheduler(action_scheduler: dict, model_name: str):
    required_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys for {model_name}: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )


def create_rewam(
    text_encoder,
    world_dit_config,
    vision_encoder,
    feature_calibration,
    temporal_bottleneck,
    camera_layout: str = "robotwin",
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    world_scheduler=None,
    action_scheduler=None,
    loss=None,
    reconstruction_probe=None,
    mot_checkpoint_mixed_attn: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.rewam import ReWAM
    from .models.text_encoder.loading import load_text_components

    world_dit_config = as_plain_dict(
        world_dit_config,
        required=True,
    )
    action_dit_config = as_plain_dict(action_dit_config)
    world_scheduler = as_plain_dict(world_scheduler)
    action_scheduler = as_plain_dict(action_scheduler, required=True)
    loss = as_plain_dict(loss)
    temporal_bottleneck = as_plain_dict(temporal_bottleneck, required=True)
    _validate_action_scheduler(action_scheduler, "ReWAM")
    proprio_dim = None if proprio_dim is None else int(proprio_dim)

    text_components = load_text_components(
        **as_plain_dict(text_encoder, required=True),
        device=device,
        torch_dtype=model_dtype,
        load_text_encoder=bool(load_text_encoder),
    )

    model = ReWAM.from_config(
        world_dit_config=world_dit_config,
        action_dit_config=action_dit_config,
        vision_encoder=as_plain_dict(vision_encoder, required=True),
        feature_calibration=as_plain_dict(feature_calibration),
        camera_layout=camera_layout,
        temporal_bottleneck=temporal_bottleneck,
        reconstruction_probe=reconstruction_probe,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        text_encoder=text_components.text_encoder,
        tokenizer=text_components.tokenizer,
        text_dim=int(world_dit_config["text_dim"]),
        proprio_dim=proprio_dim,
        device=device,
        torch_dtype=model_dtype,
        world_train_shift=world_scheduler.get("train_shift", 5.0),
        world_infer_shift=world_scheduler.get("infer_shift", 5.0),
        world_shift_base_dim=int(world_scheduler.get("shift_base_dim", 4096)),
        world_shift_scope=world_scheduler.get("shift_scope", "future_tokens"),
        world_num_train_timesteps=int(world_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_world=float(loss.get("lambda_world", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )
    model.model_paths.update(
        {
            "text_encoder": text_components.text_encoder_path,
            "tokenizer": text_components.tokenizer_path,
        }
    )
    return model


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("dataset_norm_stats")
        default_stats_path = os.path.join(work_dir.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("dataset_norm_stats")
        dataset_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with dataset_norm_stats: %s", dataset_norm_stats)
        val_ds = instantiate(data_cfg.val, dataset_norm_stats=dataset_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def train(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    work_dir.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    # Every rank constructs the same initial model; the trainer later applies
    # rank-specific seeds for data sampling and training noise.
    set_seed(int(cfg.seed))
    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = ReWAMTrainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()
