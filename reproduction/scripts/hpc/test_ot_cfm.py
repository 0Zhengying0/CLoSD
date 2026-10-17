"""CPU checks for exact coupling, condition alignment, inversion and gradients."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from closd.diffusion_planner.diffusion.ot_cfm import (
    OptimalTransportConditionalFlowMatching, couple_noise_to_data,
)


class RecordingModel(torch.nn.Module):
    all_goal_joint_names = ("pelvis",)

    def __init__(self, exact_velocity=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self.exact_velocity = exact_velocity

    def forward(self, x, t, y):
        self.seen_y = y
        self.seen_x = x.detach().clone()
        value = x if self.exact_velocity is None else self.exact_velocity
        return self.weight * value


class Tests(unittest.TestCase):
    def setUp(self):
        self.data = torch.tensor([[[[10.]]], [[[20.]]], [[[30.]]]])
        self.noise = torch.tensor([[[[29.]]], [[[9.]]], [[[19.]]]])
        self.y = {"mask": torch.ones(3, 1, 1, 1, dtype=torch.bool),
                  "text": ["ten", "twenty", "thirty"],
                  "prefix": torch.tensor([10., 20., 30.]),
                  "target_cond": torch.tensor([10., 20., 30.]),
                  "lengths": torch.ones(3, dtype=torch.long),
                  "target_joint_names": [["pelvis"]] * 3,
                  "is_heading": torch.zeros(3, dtype=torch.bool)}

    def test_noise_is_permuted_while_data_conditions_remain_aligned(self):
        coupled = couple_noise_to_data(self.noise, self.data)
        torch.testing.assert_close(coupled.flatten(), torch.tensor([9., 19., 29.]))
        model = RecordingModel()
        objective = OptimalTransportConditionalFlowMatching(lambda_target_loc=0)
        loss = objective.training_losses(model, self.data, torch.zeros(3, dtype=torch.long),
                                         model_kwargs={"y": self.y}, noise=self.noise)
        torch.testing.assert_close(model.seen_x, coupled)
        self.assertIs(model.seen_y, self.y)
        self.assertEqual(model.seen_y["text"], ["ten", "twenty", "thirty"])
        self.assertTrue(torch.isfinite(loss["loss"]).all())
        loss["loss"].mean().backward()
        self.assertTrue(torch.isfinite(model.weight.grad))
        self.assertGreater(abs(model.weight.grad.item()), 0)

    def test_straight_path_reconstruction_matches_original_data(self):
        coupled = couple_noise_to_data(self.noise, self.data)
        model = RecordingModel(exact_velocity=self.data - coupled)
        model.weight.data.fill_(1.0)
        objective = OptimalTransportConditionalFlowMatching(lambda_target_loc=1)

        def capture_goal(pred_x, *_args):
            torch.testing.assert_close(pred_x, self.data)
            return self.y["target_cond"]

        with patch("closd.diffusion_planner.diffusion.ot_cfm.get_target_location", capture_goal), \
             patch("closd.diffusion_planner.diffusion.ot_cfm.masked_goal_l2",
                   lambda pred, ref, *_: torch.zeros(3)):
            loss = objective.training_losses(
                model, self.data, torch.tensor([0, 500, 999]),
                model_kwargs={"y": self.y}, noise=self.noise,
                dataset=SimpleNamespace(mean_gpu=None, std_gpu=None,
                                        t2m_dataset=SimpleNamespace(
                                            opt=SimpleNamespace(joints_num=22))))
        torch.testing.assert_close(loss["loss"], torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
