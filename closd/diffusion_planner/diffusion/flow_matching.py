import math
import torch

from closd.diffusion_planner.utils.loss_util import (
    masked_l2,
    masked_goal_l2,
)
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import (
    get_target_location,
)


class FlowMatching:
    """
    Flow Matching for DiP motion prediction.

    tau = 0 : Gaussian noise
    tau = 1 : motion data

    Path:
        x_tau = cos(pi/2 * tau) * eps
              + sin(pi/2 * tau) * x_data
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
        to continuous tau in [0, 1].
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
        """
        x_start:
            ground-truth future motion [B, J, F, T]

        model output:
            velocity field v_theta
        """

        if model_kwargs is None:
            model_kwargs = {}

        y = model_kwargs["y"]

        # Same valid-frame mask used by DiP.
        mask = y["mask"]

        if "keyframe_mask" in y:
            mask = torch.logical_and(mask, ~y["keyframe_mask"])

        # Source distribution: Gaussian noise.
        if noise is None:
            noise = torch.randn_like(x_start)

        # Integer timestep -> continuous time tau in [0, 1].
        tau = self._continuous_time(t, x_start)

        theta = 0.5 * math.pi * tau

        c = torch.cos(theta)
        s = torch.sin(theta)

        # -------------------------------------------------
        # 1. Construct a point on the Flow Matching path.
        #
        # tau=0 -> noise
        # tau=1 -> real motion
        # -------------------------------------------------
        x_t = c * noise + s * x_start

        # -------------------------------------------------
        # 2. Ground-truth velocity of this probability path.
        # -------------------------------------------------
        k = 0.5 * math.pi

        target_v = k * (-s * noise + c * x_start)

        # -------------------------------------------------
        # 3. MDM now predicts velocity rather than x_start.
        # -------------------------------------------------
        pred_v = model(
            x_t,
            t,
            **model_kwargs,
        )

        terms = {}

        terms["flow_mse"] = masked_l2(
            target_v,
            pred_v,
            mask,
        )

        # -------------------------------------------------
        # 4. Recover predicted x_start from (x_t, pred_v).
        #
        # x_start_hat =
        #     sin(theta) * x_t
        #   + cos(theta) * pred_v / (pi/2)
        #
        # This lets us keep DiP's target-location loss.
        # -------------------------------------------------
        pred_x_start = s * x_t + c * (pred_v / k)

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
            terms["flow_mse"]
            + self.lambda_target_loc * terms.get("target_loc", 0.0)
        )

        return terms