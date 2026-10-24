"""Local, read-only LeRobot datasets with v2 storage adapted to the v3 reader."""

import warnings
from bisect import bisect_right
from pathlib import Path

import datasets
import jsonlines
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as _LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
    hf_transform_to_torch,
    load_info,
    load_stats,
)
from lerobot.datasets.video_utils import (
    VideoDecoderCache,
    decode_video_frames,
    decode_video_frames_torchcodec,
    get_safe_default_codec,
)


class LeRobotDatasetMetadata(_LeRobotDatasetMetadata):
    """Expose local v2/v3 metadata through the upstream reader's v3 interface."""

    def __init__(self, repo_id: str, root: str | Path):
        self.repo_id = repo_id
        self.root = Path(root)
        self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        self.is_v2 = self.info["codebase_version"] in {"v2.0", "v2.1"}
        if self.is_v2:
            with jsonlines.open(self.root / "meta/tasks.jsonl") as reader:
                tasks = {int(row["task_index"]): row["task"] for row in reader}
            task_ids = sorted(tasks)
            self.tasks = pd.DataFrame({"task_index": task_ids}, index=[tasks[i] for i in task_ids])
            with jsonlines.open(self.root / "meta/episodes.jsonl") as reader:
                episodes = {int(row["episode_index"]): row for row in reader}
            start = 0
            for episode_id in sorted(episodes):
                episode = episodes[episode_id]
                episode["dataset_from_index"] = start
                start += int(episode["length"])
                episode["dataset_to_index"] = start
                for key in self.video_keys:
                    episode[f"videos/{key}/from_timestamp"] = 0.0
            self.episodes = episodes
            self.subtasks = None
            # ReWAM computes its own normalization statistics. Do not aggregate
            # the potentially large v2 episodes_stats.jsonl just to read frames.
            self.stats = load_stats(self.root)
        elif self.info["codebase_version"] == "v3.0":
            super().load_metadata()
            self.episodes = {int(row["episode_index"]): row for row in self.episodes}
            self.tasks = self.tasks.sort_values("task_index")
        else:
            raise ValueError(f"Unsupported LeRobot dataset format: {self.info['codebase_version']}.")

    def get_data_file_path(self, ep_index: int) -> Path:
        if self.is_v2:
            return Path(
                self.data_path.format(episode_chunk=ep_index // self.chunks_size, episode_index=ep_index)
            )
        episode = self.episodes[ep_index]
        return Path(
            self.data_path.format(
                chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"]
            )
        )

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        if self.is_v2:
            return Path(
                self.video_path.format(
                    episode_chunk=ep_index // self.chunks_size, episode_index=ep_index, video_key=vid_key
                )
            )
        episode = self.episodes[ep_index]
        return Path(
            self.video_path.format(
                video_key=vid_key,
                chunk_index=episode[f"videos/{vid_key}/chunk_index"],
                file_index=episode[f"videos/{vid_key}/file_index"],
            )
        )


class LeRobotDataset(_LeRobotDataset):
    """Reuse upstream temporal queries, padding, transforms, and video decoding.

    Initialization is local and read-only because upstream's constructor fixes
    the metadata class and may download missing data. No global monkey patch is
    needed; the original dataset files and metadata remain untouched.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        *,
        metadata: LeRobotDatasetMetadata | None = None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root)
        self.meta = metadata if metadata is not None else LeRobotDatasetMetadata(repo_id, self.root)
        selected = list(self.meta.episodes if episodes is None else episodes)
        self.episodes = selected if self.meta.is_v2 else sorted(selected)
        if not self.episodes or len(self.episodes) != len(set(self.episodes)):
            raise ValueError("Select at least one episode, without duplicate episode IDs.")
        if not set(self.episodes).issubset(self.meta.episodes):
            raise ValueError("Selected episode IDs are missing from dataset metadata.")
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend or get_safe_default_codec()
        self._lazy_loading = False
        self.hf_dataset = self.load_hf_dataset()
        # Project once: select_columns rebuilds metadata for every Arrow shard.
        # Repeating it per episode makes full-dataset statistics quadratic in
        # the number of episode shards, even though each read is a small slice.
        nonvisual_keys = [key for key in self.features if key not in self.meta.camera_keys]
        self._nonvisual_hf_dataset = self.hf_dataset.select_columns(nonvisual_keys)
        lengths = [int(self.meta.episodes[i]["length"]) for i in self.episodes]
        ends = torch.tensor(lengths, dtype=torch.long).cumsum(0)
        self.episode_data_index = {"from": ends - torch.tensor(lengths), "to": ends}
        if int(ends[-1]) != len(self.hf_dataset):
            raise ValueError("Episode metadata lengths do not match the selected parquet rows.")
        # Upstream maps absolute frame indices to positions in episode subsets.
        # A dense integer array avoids a Python dict with millions of entries.
        indices = self.hf_dataset.select_columns(["index"]).with_format("numpy")[:]["index"]
        self._absolute_to_relative_idx = None
        if not np.array_equal(indices, np.arange(len(indices))):
            self._absolute_to_relative_idx = np.full(int(indices.max()) + 1, -1, dtype=np.int64)
            self._absolute_to_relative_idx[indices] = np.arange(len(indices))
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.fps)

    def load_hf_dataset(self) -> datasets.Dataset:
        if self.meta.is_v2:
            paths = [str(self.root / self.meta.get_data_file_path(i)) for i in self.episodes]
            data = datasets.Dataset.from_parquet(paths, features=get_hf_features_from_features(self.features))
        else:
            data = super().load_hf_dataset().sort(["episode_index", "frame_index"])
        data.set_transform(hf_transform_to_torch)
        return data

    def _query_videos(self, query_timestamps, ep_idx):
        if self.video_backend != "torchcodec":
            return super()._query_videos(query_timestamps, ep_idx)
        # Upstream 0.4.4 keeps every decoder/file handle in a process-wide cache.
        # Scope it to one sample, matching the original FastWAM reader's lifetime.
        cache = VideoDecoderCache()
        try:
            result = {}
            episode = self.meta.episodes[ep_idx]
            for key, timestamps in query_timestamps.items():
                path = self.root / self.meta.get_video_file_path(ep_idx, key)
                timestamps = [episode[f"videos/{key}/from_timestamp"] + t for t in timestamps]
                try:
                    frames = decode_video_frames_torchcodec(
                        path, timestamps, self.tolerance_s, decoder_cache=cache
                    )
                except Exception as error:
                    warnings.warn(f"TorchCodec decode failed ({error}); falling back to PyAV.", stacklevel=2)
                    frames = decode_video_frames(path, timestamps, self.tolerance_s, backend="pyav")
                result[key] = frames.squeeze(0)
            return result
        finally:
            cache.clear()

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for {len(self)} frames.")
        return super().__getitem__(idx)

    def get_episode_data(self, episode_idx: int):
        """Read nonvisual columns for ReWAM statistics, without decoding video."""
        start = int(self.episode_data_index["from"][episode_idx])
        end = int(self.episode_data_index["to"][episode_idx])
        data = self._nonvisual_hf_dataset[start:end]
        return {
            key: torch.stack(values) if values and isinstance(values[0], torch.Tensor) else np.asarray(values)
            for key, values in data.items()
        }


class MultiLeRobotDataset(torch.utils.data.ConcatDataset):
    """Concatenate local readers while preserving dataset and episode indices."""

    def __init__(self, metadata, episodes, delta_timestamps, tolerance_s=None, video_backend=None):
        super().__init__(
            [
                LeRobotDataset(
                    meta.repo_id,
                    root=meta.root,
                    metadata=meta,
                    episodes=episodes[meta.repo_id],
                    delta_timestamps=delta_timestamps,
                    tolerance_s=1e-4 if tolerance_s is None else tolerance_s,
                    video_backend=video_backend,
                )
                for meta in metadata
            ]
        )
        self.disabled_features = set.union(*(set(ds.features) for ds in self.datasets)) - set.intersection(
            *(set(ds.features) for ds in self.datasets)
        )

    @property
    def num_frames(self):
        return len(self)

    @property
    def num_episodes(self):
        return sum(ds.num_episodes for ds in self.datasets)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for {len(self)} frames.")
        item = super().__getitem__(idx)
        item["dataset_index"] = torch.tensor(bisect_right(self.cumulative_sizes, idx))
        for key in self.disabled_features:
            item.pop(key, None)
        return item

    def get_episode_data(self, episode_idx):
        for dataset in self.datasets:
            if episode_idx < dataset.num_episodes:
                return dataset.get_episode_data(episode_idx)
            episode_idx -= dataset.num_episodes
        raise IndexError("Episode index out of bounds.")
