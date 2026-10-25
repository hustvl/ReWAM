from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import FRAMEWORK_REGISTRY

from .data import SamplingPlan
from .model import ReWAM


class ReWAMFramework(ReWAM, baseframework):
    """StarVLA training and serving contracts backed by the independent model."""

    @property
    def observation_history_stride(self) -> int:
        """Expose sampling metadata to the host's observation history buffer."""
        return int(self.config.datasets.vla_data.observation_stride)

    def predict_action(self, examples, **kwargs):
        # The host adapter owns default evaluation lengths; the network only
        # receives the requested lengths and actual observation tensors.
        if "action_horizon" not in kwargs:
            kwargs["action_horizon"] = int(self.config.datasets.vla_data.action_horizon)
        if self.variant == "joint" and "num_future_frames" not in kwargs:
            data = self.config.datasets.vla_data
            plan = SamplingPlan(
                int(kwargs["action_horizon"]), int(data.observation_stride), int(data.num_context_frames)
            )
            kwargs["num_future_frames"] = plan.num_future_frames
        return super().predict_action(examples, **kwargs)


def register() -> None:
    for name in ("ReWAM", "ReWAMJoint"):
        existing = FRAMEWORK_REGISTRY._registry.get(name)
        if existing is not None and existing is not ReWAMFramework:
            raise RuntimeError(
                f"Framework {name!r} is already registered by {existing}; remove the legacy ReWAM implementation before enabling the plugin."
            )
        FRAMEWORK_REGISTRY.register(name)(ReWAMFramework)
