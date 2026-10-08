import torch

from closd.diffusion_planner.utils.loss_util import (
    masked_l2,
    masked_goal_l2,
)
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import (
    get_target_location,
)


class RectifiedFlow:
    """
    1-Rectified Flow for DiP.

    t = 0: Gaussian noise
    t = 1: motion data

    Straight interpolation:
        x_t = (1 - t) * noise + t * x_data

    Target velocity:
        v = x_data - noise
    """

    def __init__(
        self,
        num_timesteps=1000,
        lambda_target_loc=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.lambda_target_loc = lambda_target_loc

    def _continuous_time(self, t, x):
        """
        Convert integer timestep indices [0, T-1]
        to continuous t in [0, 1].
        """
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

        mask = y["mask"]

        if "keyframe_mask" in y:
            mask = torch.logical_and(
                mask,
                ~y["keyframe_mask"],
            )

        # Gaussian source sample.
        if noise is None:
            noise = torch.randn_like(x_start)

        tau = self._continuous_time(t, x_start)

        # -------------------------------------------------
        # 1. Straight Rectified Flow interpolation.
        #
        # t = 0 -> noise
        # t = 1 -> data
        # -------------------------------------------------
        x_t = (
            (1.0 - tau) * noise
            + tau * x_start
        )

        # -------------------------------------------------
        # 2. Exact velocity along the straight path.
        # -------------------------------------------------
        target_v = x_start - noise

        # -------------------------------------------------
        # 3. MDM predicts velocity.
        # -------------------------------------------------
        pred_v = model(
            x_t,
            t,
            **model_kwargs,
        )

        terms = {}

        terms["rf_mse"] = masked_l2(
            target_v,
            pred_v,
            mask,
        )

        # -------------------------------------------------
        # 4. Recover predicted clean motion:
        #
        # x_data = x_t + (1-t) * v
        # -------------------------------------------------
        pred_x_start = (
            x_t
            + (1.0 - tau) * pred_v
        )

        # -------------------------------------------------
        # 5. Keep DiP target-location supervision.
        # -------------------------------------------------
        if self.lambda_target_loc > 0.0:
            ref_target = y["target_cond"]

            pred_target = get_target_location(
                pred_x_start,
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
                ref_target,
                y,
                model.all_goal_joint_names,
            )

        terms["loss"] = (
            terms["rf_mse"]
            + self.lambda_target_loc
            * terms.get("target_loc", 0.0)
        )

        return terms
