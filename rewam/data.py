from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingPlan:
    action_horizon: int
    observation_stride: int
    num_context_frames: int

    def __post_init__(self):
        if min(self.action_horizon, self.observation_stride, self.num_context_frames) <= 0:
            raise ValueError("Sampling lengths and observation_stride must be positive.")
        if self.action_horizon % (self.observation_stride * self.num_context_frames):
            raise ValueError("action_horizon must be divisible by observation_stride * num_context_frames.")

    @property
    def num_future_frames(self) -> int:
        return self.action_horizon // self.observation_stride

    @property
    def history_offsets(self) -> list[int]:
        return list(
            range(-(self.num_context_frames - 1) * self.observation_stride, 0, self.observation_stride)
        )

    @property
    def future_offsets(self) -> list[int]:
        return list(range(self.observation_stride, self.action_horizon + 1, self.observation_stride))

    @property
    def image_offsets(self) -> list[int]:
        return [*self.history_offsets, 0, *self.future_offsets]
