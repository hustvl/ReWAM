import inspect
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rewam.datasets.processor import ReWAMProcessor  # noqa: E402
from rewam.datasets.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from rewam.datasets.utils import load_dataset_stats_from_json  # noqa: E402

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
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


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}") from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _load_sim_cfg(
    resolved_sim_cfg_path: Optional[str],
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    if _is_none_like(resolved_sim_cfg_path):
        return _compose_sim_cfg(
            sim_cfg_path=sim_cfg_path,
            sim_cfg_name=sim_cfg_name,
            sim_task=sim_task,
        )

    config_path = Path(str(resolved_sim_cfg_path)).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Resolved simulation config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"Resolved simulation config must be a mapping, got {type(cfg).__name__}: " f"{config_path}"
        )
    logger.info("Loaded resolved simulation config: %s", config_path)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. " "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _get_num_video_frames(cfg: DictConfig) -> int:
    """Return source frames expected by inference, including TRB history."""
    horizon = int(cfg.data.train.action_horizon)
    stride = int(cfg.data.train.observation_stride)
    context_frames = int(cfg.data.train.num_context_frames)
    if horizon <= 0 or stride <= 0 or horizon % stride:
        raise ValueError("data.train.action_horizon must be positive and divisible by observation_stride.")
    if context_frames <= 0 or (horizon // stride) % context_frames:
        raise ValueError("Future image frames must be divisible by num_context_frames.")
    return horizon // stride + context_frames


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        observation_stride: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        full_obs_episode_probability: float,
        visualize_future_video: bool,
        future_video_max_episodes: int,
        eval_output_dir: Path,
        task_config: str,
        compile_action_infer: bool = False,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: ReWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.observation_stride = int(observation_stride)
        if self.observation_stride <= 0:
            raise ValueError("observation_stride must be positive.")
        self.num_history_frames = self.model.num_history_frames
        self.num_inference_steps = int(num_inference_steps)
        self.compile_action_infer = bool(compile_action_infer)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        validate_timeline = getattr(self.model, "validate_inference_timeline", None)
        if callable(validate_timeline):
            validate_timeline(
                num_video_frames=self._num_video_frames,
                action_horizon=self.action_horizon,
            )
        self.full_obs_episode_probability = float(full_obs_episode_probability)
        if not 0.0 <= self.full_obs_episode_probability <= 1.0:
            raise ValueError(
                "full_obs_episode_probability must be within [0, 1], "
                f"got {self.full_obs_episode_probability}."
            )
        self._observation_rng = np.random.default_rng(seed)
        self._full_obs_episode = False
        self.visualize_future_video = bool(visualize_future_video)
        self.future_video_max_episodes = int(future_video_max_episodes)
        self.future_video_dir = Path(eval_output_dir) / "future_rollouts" / str(task_config)
        if self.visualize_future_video:
            if self.future_video_max_episodes <= 0:
                raise ValueError("future_video_max_episodes must be positive when visualization is enabled.")
            if self.model.reconstruction_probe is None:
                raise ValueError(
                    "Future RGB visualization requires the detached feature reconstruction probe."
                )
            if not callable(getattr(self.model, "infer_joint", None)):
                raise ValueError(
                    "visualize_future_video=true requires the configured model to implement infer_joint()."
                )
            self.future_video_dir.mkdir(parents=True, exist_ok=True)

        self.pending_actions: deque[np.ndarray] = deque()
        self._history_images: dict[int, torch.Tensor] = {}
        self.episode_count = 0
        self.step_count = 0
        self.replan_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | "
            "replan=%d | full_obs_episode_probability=%.3f",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.full_obs_episode_probability,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        self.processor.validate_action_state({"state": state_tensor})
        return self.processor.normalizer.normalizers["state"].forward(state_tensor)

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        normalizer = self.processor.normalizer.normalizers["action"]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _save_future_rollout(self, frames: list[Any]) -> None:
        if len(frames) == 0:
            raise ValueError("Cannot save an empty future rollout.")
        arrays = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            else:
                array = np.asarray(frame, dtype=np.uint8)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Future rollout frame must be RGB, got {array.shape}.")
            arrays.append(array)
        target_height = arrays[0].shape[0]
        arrays = [
            (
                array
                if array.shape[0] == target_height
                else _resize_rgb(
                    array,
                    (
                        int(round(array.shape[1] * target_height / array.shape[0])),
                        target_height,
                    ),
                )
            )
            for array in arrays
        ]
        strip = np.concatenate(arrays, axis=1)
        episode_dir = self.future_video_dir / f"episode_{self.episode_count - 1:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        output_path = episode_dir / (f"replan_{self.replan_count:03d}_step_{self.step_count:06d}.png")
        Image.fromarray(strip, mode="RGB").save(output_path)
        logger.info("Saved predicted future rollout strip to %s", output_path)

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        self._history_images[self.step_count] = image_tensor
        history_steps = [
            max(0, self.step_count - offset * self.observation_stride)
            for offset in range(self.num_history_frames, 0, -1)
        ]
        history_images = (
            torch.stack([self._history_images[step] for step in history_steps], dim=2)
            if history_steps
            else image_tensor.unsqueeze(2)[:, :, :0]
        )
        oldest_step = max(0, self.step_count - self.num_history_frames * self.observation_stride)
        self._history_images = {
            step: image for step, image in self._history_images.items() if step >= oldest_step
        }
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_kwargs["history_images"] = history_images
        capture_future = (
            self.visualize_future_video
            and self.episode_count > 0
            and self.episode_count <= self.future_video_max_episodes
        )
        if capture_future or "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            if capture_future:
                joint_kwargs = dict(infer_kwargs)
                if "test_action_with_infer_action" in inspect.signature(self.model.infer_joint).parameters:
                    joint_kwargs["test_action_with_infer_action"] = False
                pred = self.model.infer_joint(**joint_kwargs)
            else:
                infer_kwargs["compile_action_infer"] = self.compile_action_infer
                pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        if capture_future:
            self._save_future_rollout(pred["video"])
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        self.replan_count += 1
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def _needs_history_observation(self) -> bool:
        # Capture observations needed by any later replan, including when a
        # history window spans several action chunks or strides do not divide chunks.
        return any(
            (self.step_count + offset * self.observation_stride) % self.replan_steps == 0
            for offset in range(1, self.num_history_frames + 1)
        )

    def should_request_observation(self) -> bool:
        return self._full_obs_episode or not self.pending_actions or self._needs_history_observation()

    def get_observation_refresh_mode(self) -> str:
        if self._full_obs_episode:
            return "full"
        return "replan+history" if self.num_history_frames else "replan-only"

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if self._needs_history_observation():
            if observation is None:
                raise ValueError("Observation is required at a TRB history refresh step.")
            self._history_images[self.step_count] = self._build_robotwin_image_tensor(observation)
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty " "(replan step for rewam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self._history_images.clear()
        self.episode_count += 1
        self.step_count = 0
        self.replan_count = 0
        self._full_obs_episode = bool(self._observation_rng.random() < self.full_obs_episode_probability)
        logger.info(
            "Episode %d observation refresh mode: %s",
            self.episode_count - 1,
            self.get_observation_refresh_mode(),
        )
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    resolved_sim_cfg_path = usr_args.get("resolved_sim_cfg_path")
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _load_sim_cfg(
        resolved_sim_cfg_path=resolved_sim_cfg_path,
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.action_horizon)
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False)))
    full_obs_episode_probability = float(
        usr_args.get(
            "full_obs_episode_probability",
            cfg.EVALUATION.get("full_obs_episode_probability", 0.0),
        )
    )
    visualize_future_video = _parse_bool(
        usr_args.get(
            "visualize_future_video",
            cfg.EVALUATION.get("visualize_future_video", False),
        )
    )
    future_video_max_episodes = int(
        usr_args.get(
            "future_video_max_episodes",
            cfg.EVALUATION.get("future_video_max_episodes", 1),
        )
    )
    eval_output_dir_value = usr_args.get("eval_output_dir")
    if _is_none_like(eval_output_dir_value):
        eval_output_dir_value = cfg.EVALUATION.get("output_dir", ".")
    eval_output_dir = Path(str(eval_output_dir_value)).expanduser().resolve()
    task_config = str(usr_args.get("task_config", cfg.EVALUATION.get("task_config", "unknown")))

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        observation_stride=int(cfg.data.train.observation_stride),
        num_inference_steps=num_inference_steps,
        compile_action_infer=_parse_bool(
            usr_args.get("compile_action_infer", cfg.EVALUATION.get("compile_action_infer", False))
        ),
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=_get_num_video_frames(cfg),
        full_obs_episode_probability=full_obs_episode_probability,
        visualize_future_video=visualize_future_video,
        future_video_max_episodes=future_video_max_episodes,
        eval_output_dir=eval_output_dir,
        task_config=task_config,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
