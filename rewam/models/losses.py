import torch
import torch.nn.functional as F


def compute_action_flow_loss(
    *,
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: torch.Tensor | None,
    timestep_action: torch.Tensor,
    action_scheduler,
) -> torch.Tensor:
    action_loss_token = F.mse_loss(
        pred_action.float(),
        target_action.float(),
        reduction="none",
    ).mean(dim=2)
    if action_is_pad is not None:
        valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
    else:
        action_loss_per_sample = action_loss_token.mean(dim=1)
    action_weight = action_scheduler.training_weight(timestep_action).to(
        action_loss_per_sample.device,
        dtype=action_loss_per_sample.dtype,
    )
    return (action_loss_per_sample * action_weight).mean()
