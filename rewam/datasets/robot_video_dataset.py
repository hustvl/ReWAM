import hashlib
import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import Compose, ConvertImageDtype, Normalize
from tqdm import tqdm

from rewam.utils.work_dir import get_work_dir

from .lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
from .utils import compose_camera_video, load_dataset_stats_from_json, save_dataset_stats_to_json

logger = logging.getLogger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def text_embed_cache_filename(prompt: str, context_len: int) -> str:
    """Preserve the existing UMT5 cache names independently of resource locations."""
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"


class RobotVideoDataset(torch.utils.data.Dataset):
    """ReWAM sampling, normalization, and model inputs over local LeRobot readers."""

    def __init__(
        self,
        dataset_root: str | Path | Sequence[str | Path],
        shape_meta,
        action_horizon: int = 32,
        processor=None,
        text_embed_cache: str | None = None,
        tolerance_s: float | None = None,
        video_backend: str | None = None,
        context_len=128,
        dataset_norm_stats: str | None = None,
        val_set_proportion=0.05,
        is_training_set=False,
        observation_stride: int = 1,
        num_context_frames: int = 2,
        history_repeat_probability: float = 0.0,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",  # "horizontal", "vertical", "robotwin", or None
        override_instruction: str | None = None,
        seed: int = 42,
    ):
        action_horizon = int(action_horizon)
        observation_stride = int(observation_stride)
        if observation_stride <= 0:
            raise ValueError("`observation_stride` must be positive.")
        if action_horizon % observation_stride != 0:
            raise ValueError(
                "action_horizon must be divisible by observation_stride, got "
                f"{action_horizon} and {observation_stride}."
            )
        num_context_frames = int(num_context_frames)
        future_frames = action_horizon // observation_stride
        if action_horizon <= 0 or num_context_frames <= 0 or future_frames % num_context_frames:
            raise ValueError("Future video frames must be positive and divisible by num_context_frames.")
        self.num_history_frames = num_context_frames - 1
        self.image_offsets = list(
            range(-self.num_history_frames * observation_stride, action_horizon + 1, observation_stride)
        )

        self.action_horizon = action_horizon
        self.observation_stride = observation_stride
        self.is_training_set = bool(is_training_set)
        self.history_repeat_probability = float(history_repeat_probability)
        if not 0.0 <= self.history_repeat_probability <= 1.0:
            raise ValueError(
                "`history_repeat_probability` must be in [0, 1], got " f"{self.history_repeat_probability}."
            )
        self.num_future_video_transitions = action_horizon // observation_stride

        self.text_embed_cache = text_embed_cache
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.shape_meta = (
            OmegaConf.to_container(shape_meta, resolve=True)
            if OmegaConf.is_config(shape_meta)
            else deepcopy(shape_meta)
        )
        self.processor = None
        self._create_reader(dataset_root, val_set_proportion, seed, tolerance_s, video_backend)

        self.normalize_transform = Compose(
            [ConvertImageDtype(torch.float32), Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not dataset_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "dataset_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them."
                    )
                if PartialState().is_main_process:
                    logger.info("Computing dataset normalization statistics")
                    dataset_stats = self.get_dataset_stats(processor)
                    work_dir = get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                state = PartialState()
                distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
                # Validation may refer to statistics generated on rank 0's
                # local output filesystem. Read them there and distribute the
                # values, as for statistics computed from the training set.
                if state.is_main_process or not distributed:
                    dataset_stats = load_dataset_stats_from_json(dataset_norm_stats)
                    logger.info("Loaded dataset normalization statistics: path=%s", dataset_norm_stats)
                else:
                    dataset_stats = None
                if distributed:
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
                if state.is_main_process:
                    work_dir = get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.processor = processor
            if self.is_training_set:
                processor.train()
            else:
                processor.eval()

    def _create_reader(self, dataset_root, val_set_proportion, seed, tolerance_s, video_backend):
        roots = [dataset_root] if isinstance(dataset_root, (str, Path)) else dataset_root
        roots = [str(root) for root in roots]
        if not roots:
            raise ValueError("At least one dataset root is required.")
        metas = [LeRobotDatasetMetadata(repo_id=root, root=root) for root in roots]
        fps_values = {meta.fps for meta in metas}
        if len(fps_values) != 1:
            raise ValueError(f"All datasets must have the same fps, got {sorted(fps_values)}.")
        fps = metas[0].fps
        self.image_meta = self.shape_meta["images"]
        self.action_meta = self.shape_meta["action"]
        self.state_meta = self.shape_meta["state"]
        if len(self.state_meta) != 1 or len(self.action_meta) != 1:
            raise ValueError("ReWAM expects one native action vector and one native state vector.")

        delta_timestamps = {}
        fields = (
            (self.image_meta, "observation.images", self.image_offsets),
            (self.state_meta, "observation.state", range(self.action_horizon + 1)),
            (self.action_meta, "action", range(self.action_horizon)),
        )
        for metadata, prefix, offsets in fields:
            for meta in metadata:
                key = prefix if meta["key"] == "default" else f"{prefix}.{meta['key']}"
                meta["lerobot_key"] = key
                delta_timestamps[key] = [offset / fps for offset in offsets]

        episodes = {}
        for meta in metas:
            selected = sorted(meta.episodes)
            if val_set_proportion >= 1e-6:
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                np.random.default_rng(seed).shuffle(selected)
                selected = selected[:split_idx] if self.is_training_set else selected[split_idx:]
            episodes[meta.repo_id] = selected
        self.reader = MultiLeRobotDataset(
            metadata=metas,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            video_backend=video_backend,
        )
        starts, ends, offset = [], [], 0
        for dataset in self.reader.datasets:
            starts.append(dataset.episode_data_index["from"] + offset)
            ends.append(dataset.episode_data_index["to"] + offset)
            offset += len(dataset)
        self.episode_data_index = {"from": torch.cat(starts), "to": torch.cat(ends)}

    def __len__(self):
        return len(self.reader)

    @staticmethod
    def _read_vector(meta, sample, kind):
        value = sample[meta["lerobot_key"]]
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        if value.shape[-1] != meta["shape"]:
            raise ValueError(
                f"{kind.capitalize()} '{meta['key']}' dimension {value.shape[-1]} "
                f"does not match shape {meta['shape']}."
            )
        return value

    def _load_sample(self, idx):
        """Read and preprocess a temporal sample before video composition."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for {len(self)} frames.")
        sample_idx = idx
        for attempt in range(5):
            try:
                raw = self.reader[sample_idx]
                break
            except Exception:
                if attempt == 4:
                    raise RuntimeError(f"Failed to load a sample after 5 attempts for index {idx}.")
                logger.warning(
                    "Error loading sample %s; retrying with a random index.", sample_idx, exc_info=True
                )
                sample_idx = np.random.randint(len(self))
        sample = {
            "idx": sample_idx,
            "task": raw["task"],
            "action": self._read_vector(self.action_meta[0], raw, "action"),
            "state": self._read_vector(self.state_meta[0], raw, "state"),
            "images": {},
            "action_is_pad": raw[f"{self.action_meta[0]['lerobot_key']}_is_pad"],
            "state_is_pad": raw[f"{self.state_meta[0]['lerobot_key']}_is_pad"],
            "image_is_pad": raw[f"{self.image_meta[0]['lerobot_key']}_is_pad"],
        }
        for meta in self.image_meta:
            image = raw[meta["lerobot_key"]]
            if image.ndim == 3:
                image = image.unsqueeze(0)
            sample["images"][meta["key"]] = (image * 255).to(torch.uint8)
        for key in raw:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = raw[key]
        return self.processor.preprocess(sample) if self.processor is not None else sample

    def _get_episode_data(self, episode_idx):
        raw = self.reader.get_episode_data(episode_idx)
        state = self._read_vector(self.state_meta[0], raw, "state").unsqueeze(1).float()
        action = self._read_vector(self.action_meta[0], raw, "action")
        indices = torch.arange(len(action))[:, None] + torch.arange(self.action_horizon)
        return {"state": state, "action": action[indices.clamp_max(len(action) - 1)].float()}

    def get_dataset_stats(self, processor):
        """Keep the existing episode-weighted global normalization statistics."""
        moments = {kind: defaultdict(list) for kind in ("state", "action")}
        episodes_num = self.reader.num_episodes

        def process_episode(idx):
            return processor.validate_action_state(self._get_episode_data(idx))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_episode, idx) for idx in range(episodes_num)]
            for future in tqdm(as_completed(futures), total=episodes_num, desc="Calculating normalization"):
                batch = future.result()
                for kind in moments:
                    values = batch[kind]
                    statistics = {
                        "min": values.amin(0),
                        "max": values.amax(0),
                        "mean": values.mean(0),
                        "var": values.var(0),
                        "q01": torch.quantile(values, 0.01, dim=0),
                        "q99": torch.quantile(values, 0.99, dim=0),
                    }
                    for name, value in statistics.items():
                        moments[kind][name].append(value)
        stats = {"state": {}, "action": {}, "num_episodes": episodes_num, "num_transition": len(self)}
        for kind, metadata in (("state", self.state_meta), ("action", self.action_meta)):
            values = moments[kind]
            means = torch.stack(values["mean"])
            variances = torch.stack(values["var"])
            global_mean = means.mean((0, 1))
            stats[kind][metadata[0]["key"]] = {
                "global_min": torch.stack(values["min"]).amin(0).amin(0),
                "global_max": torch.stack(values["max"]).amax(0).amax(0),
                "global_q01": torch.stack(values["q01"]).amin(0).amin(0),
                "global_q99": torch.stack(values["q99"]).amax(0).amax(0),
                "global_mean": global_mean,
                "global_std": (variances + (means - global_mean) ** 2).mean((0, 1)).sqrt(),
            }
        return stats

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self._load_sample(sample_idx)

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self))

        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            T_video, C, H, W = video.shape

        if (
            self.num_history_frames > 0
            and self.is_training_set
            and self.history_repeat_probability > 0.0
            and np.random.rand() < self.history_repeat_probability
        ):
            video = video.clone()
            if video.ndim == 5:
                video[:, : self.num_history_frames] = video[
                    :, self.num_history_frames : self.num_history_frames + 1
                ]
            else:
                video[: self.num_history_frames] = video[self.num_history_frames]
            image_is_pad = image_is_pad.clone()
            image_is_pad[: self.num_history_frames] = image_is_pad[self.num_history_frames]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        video = compose_camera_video(video, self.concat_multi_camera)

        # The camera layout determines the final spatial size.
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3)  # [C, T_video, H, W], range [-1, 1]

        # Drop the terminal state so each proprioceptive input precedes its action.
        action = sample["action"]  # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :]  # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % self.num_future_video_transitions != 0:
            raise ValueError(
                "`action` horizon must be divisible by future video transitions, got "
                f"{action.shape[0]} and {self.num_future_video_transitions}."
            )

        task = sample["instruction"]

        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.text_embed_cache:
            context, context_mask = self._get_cached_text_context(instruction)
            # Keep Wan's zero-padding and all-valid context mask convention.
            context[~context_mask] = 0.0
            data["context"] = context
            data["context_mask"] = torch.ones_like(context_mask)
        return data

    def _get_cached_text_context(self, prompt: str):
        cache_dir = self.text_embed_cache
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, text_embed_cache_filename(prompt, self.context_len))
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. " "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception:
            logger.warning(
                "Failed to process dataset sample: index=%d; retrying with a random sample",
                idx,
                exc_info=True,
            )
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
