import torch

from closd.diffusion_planner.utils.loss_util import (
    masked_l2,
    masked_goal_l2,
)
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import (
    get_target_location,
)


class RectifiedFlowReflow:

    def __init__(
        self,
        num_timesteps=1000,
        lambda_target_loc=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.lambda_target_loc = lambda_target_loc

    def _continuous_time(self, t, x):
        tau = t.float() / (self.num_timesteps - 1)

        shape = [x.shape[0]] + [1] * (x.ndim - 1)

        return tau.view(*shape)

    def training_losses(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
        dataset=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        y = model_kwargs["y"]

        # In RF-2, x_start is teacher endpoint z1.
        z1 = x_start

        if "reflow_noise" not in y:
            raise KeyError(
                "RF-2 requires y['reflow_noise']."
            )

        z0 = y["reflow_noise"]

        if z0.shape != z1.shape:
            raise ValueError(
                f"z0/z1 mismatch: "
                f"{tuple(z0.shape)} vs {tuple(z1.shape)}"
            )

        mask = y["mask"]

        tau = self._continuous_time(
            t,
            z1,
        )

        # Straight reflow path.
        x_t = (
            (1.0 - tau) * z0
            + tau * z1
        )

        target_v = z1 - z0

        pred_v = model(
            x_t,
            t,
            **model_kwargs,
        )

        terms = {}

        terms["rf2_mse"] = masked_l2(
            target_v,
            pred_v,
            mask,
        )

        # Reconstruct paired endpoint z1.
        pred_z1 = (
            x_t
            + (1.0 - tau) * pred_v
        )

        if self.lambda_target_loc > 0.0:

            pred_target = get_target_location(
                pred_z1,
                dataset.mean_gpu,
                dataset.std_gpu,
                y["lengths"],
                dataset.t2m_dataset.opt.joints_num,
                model.all_goal_joint_names,
                y["target_joint_names"],
                y["is_heading"],
            )

            terms["target_loc"] = masked_goal_l2(
                pred_target,
                y["target_cond"],
                y,
                model.all_goal_joint_names,
            )

        terms["loss"] = (
            terms["rf2_mse"]
            + self.lambda_target_loc
            * terms.get("target_loc", 0.0)
        )

        return terms
