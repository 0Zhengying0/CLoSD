"""Minibatch exact OT conditional flow matching for conditional motion prediction.

The data batch stays in its original order. Only Gaussian source samples are
permuted, so text, prefix, target location, masks and timestep still describe
the same motion example after coupling.
"""

import torch
from scipy.optimize import linear_sum_assignment

from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import get_target_location
from closd.diffusion_planner.utils.loss_util import masked_goal_l2, masked_l2


def couple_noise_to_data(noise, data):
    """Return exact equal-mass minibatch OT sources in data batch order.

    The squared Euclidean assignment is the integral extreme point of the
    discrete OT linear program used by the official exact OT matcher. Each
    source is used once, preserving the empirical Gaussian source marginal.
    """
    if noise.shape != data.shape or noise.ndim < 2:
        raise ValueError("noise and data must have the same batched shape")
    with torch.no_grad():
        source = noise.detach().float().flatten(1)
        target = data.detach().float().flatten(1)
        # cdist avoids a [batch, batch, motion_dimension] temporary tensor.
        cost = torch.cdist(source, target).square().cpu().numpy()
        source_idx, target_idx = linear_sum_assignment(cost)
        permutation = torch.empty(noise.shape[0], dtype=torch.long)
        permutation[torch.as_tensor(target_idx)] = torch.as_tensor(source_idx)
    return noise.index_select(0, permutation.to(noise.device))


class OptimalTransportConditionalFlowMatching:
    """Straight, zero-path-noise OT-CFM with the original MDM and goal loss."""

    def __init__(self, num_timesteps=1000, lambda_target_loc=1.0):
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be >= 2")
        self.num_timesteps = num_timesteps
        self.lambda_target_loc = lambda_target_loc

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dataset=None):
        if model_kwargs is None:
            model_kwargs = {}
        y = model_kwargs["y"]
        mask = y["mask"]
        if "keyframe_mask" in y:
            mask = torch.logical_and(mask, ~y["keyframe_mask"])
        if noise is None:
            noise = torch.randn_like(x_start)
        noise = couple_noise_to_data(noise, x_start)
        tau = (t.float() / (self.num_timesteps - 1)).view(
            x_start.shape[0], *([1] * (x_start.ndim - 1))
        )
        x_t = (1.0 - tau) * noise + tau * x_start
        target_v = x_start - noise
        pred_v = model(x_t, t, **model_kwargs)
        terms = {"ot_cfm_mse": masked_l2(target_v, pred_v, mask)}

        if self.lambda_target_loc > 0.0:
            # For a straight path x1 = xt + (1-t)u. The VP-CFM rotation
            # inverse is invalid here, including at the path endpoints.
            pred_x_start = x_t + (1.0 - tau) * pred_v
            pred_target = get_target_location(
                pred_x_start, dataset.mean_gpu, dataset.std_gpu,
                y["lengths"], dataset.t2m_dataset.opt.joints_num,
                model.all_goal_joint_names, y["target_joint_names"],
                y["is_heading"],
            )
            terms["target_loc"] = masked_goal_l2(
                pred_target, y["target_cond"], y, model.all_goal_joint_names
            )
        terms["loss"] = terms["ot_cfm_mse"] + self.lambda_target_loc * terms.get("target_loc", 0.0)
        return terms
