from dataclasses import asdict, dataclass, field

from omegaconf import OmegaConf


def merge_framework_config(default_config_cls, config):
    """Fill framework defaults while keeping the host's dataset/trainer config."""
    config.framework = OmegaConf.merge(asdict(default_config_cls()), config.get("framework", {}))
    return config


def as_thw(value) -> tuple[int, int, int]:
    """Validate temporal/height/width patch sizes."""
    if len(value) != 3 or any(int(size) <= 0 or int(size) != size for size in value):
        raise ValueError(f"Expected positive integer patch_size [T,H,W], got {value}.")
    return tuple(int(size) for size in value)


@dataclass
class ReWAMDefaultConfig:
    """Defaults for the ReWAM Base architecture."""

    name: str = "ReWAM"
    variant: str = "base"
    compile_action_infer: bool = True
    vision_encoder: dict = field(
        default_factory=lambda: {
            "model_name": "dinov3_vitl16",
            "hub_repo": "facebookresearch/dinov3:main",
            "hub_source": "github",
            "hub_kwargs": {},
            "output_dim": 1024,
        }
    )
    feature_calibration: dict = field(default_factory=lambda: {"layers": [11, 13, 15, 17, 19, 21, 23]})
    temporal_bottleneck: dict = field(
        default_factory=lambda: {
            "output_dim": 128,
            "patch_size": [2, 2, 2],
            "gradient_mode": "action_only",
            "qk_dim": 1024,
            "num_heads": 8,
            "norm_eps": 1e-6,
            "rope_base": 10000.0,
            "position_scale": 16.0,
            "causal_normalization": {
                "eps": 1e-5,
                "momentum": 0.1,
            },
        }
    )
    reconstruction_probe: dict | None = field(
        default_factory=lambda: {"loss_weight": 1.0, "decoder_path": None}
    )
    world_dit_config: dict = field(
        default_factory=lambda: {
            "hidden_dim": 2048,
            "ffn_dim": 8192,
            "freq_dim": 256,
            "text_dim": 4096,
            "num_heads": 24,
            "attn_head_dim": 128,
            "num_layers": 30,
            "eps": 1e-6,
            "patch_size": [1, 1, 1],
        }
    )
    action_dit_config: dict = field(
        default_factory=lambda: {
            "action_dim": 14,
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "text_dim": 4096,
            "freq_dim": 256,
            "num_heads": 24,
            "attn_head_dim": 128,
            "num_layers": 30,
            "eps": 1e-6,
        }
    )
    load_text_encoder: bool = False
    text_encoder: dict = field(
        default_factory=lambda: {
            "pretrained_path": None,
            "max_length": 128,
            "prompt_template": "A video recorded from a robot's point of view executing the following instruction: {lang}",
        }
    )
    mot_checkpoint_mixed_attn: bool = False
    world_scheduler: dict = field(
        default_factory=lambda: {
            "train_shift": "auto",
            "infer_shift": "auto",
            "shift_base_dim": 4096,
            "shift_scope": "future_tokens",
            "num_train_timesteps": 1000,
        }
    )
    action_scheduler: dict = field(
        default_factory=lambda: {
            "train_shift": 5.0,
            "infer_shift": 5.0,
            "num_train_timesteps": 1000,
            "num_inference_steps": 20,
        }
    )
    loss: dict = field(default_factory=lambda: {"lambda_world": 1.0, "lambda_action": 1.0})
    proprio_dim: int | None = 14
