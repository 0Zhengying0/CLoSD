"""CPU-only capture, geometry, checkpoint-args, summary and media smoke tests."""

import contextlib
import io
import json
import joblib
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from closd.learning.quality_snapshot import QualitySnapshotCapture, TASKS
from closd.learning.quality_execution_record import QualityExecutionRecorder
from closd.utils.closd_util import STATES
from closd.utils.poselib.poselib.skeleton.skeleton3d import SkeletonTree
from quality_metrics import metrics, recover_xyz
from quality_render import render_video, verify_media
from run_dip_fm_quality import load_cases, sample_one, validate_models
import run_dip_fm_quality as quality_run
import summarize_dip_fm_quality as summary
from prepare_official_sim_record import prepare
from render_quality_execution import select_episodes


class QualityTests(unittest.TestCase):
    def test_official_sim_preparation_and_pair_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = SkeletonTree.from_mjcf(str(ROOT / "closd/data/assets/mjcf/smpl_humanoid.xml"))
            count = len(tree.node_names)
            self.assertEqual(count, 24)
            quats = torch.zeros(4, count, 4)
            quats[..., 3] = 1
            sample = {"skeleton_tree": tree.to_dict(), "body_quat": quats,
                      "trans": torch.tensor([[0., 0., 1.]] * 4),
                      "ref_body_pos_full": torch.zeros(4, count, 3),
                      "target_trans": torch.zeros(4, 3),
                      "target_quat": quats[:, 0],
                      "task_state": torch.tensor([int(STATES.SIT), int(STATES.SIT),
                                                   int(STATES.GET_UP), int(STATES.GET_UP)]),
                      "init_state": torch.tensor(int(STATES.SIT)), "fps": 60}
            record = {"0_0": sample}
            self.assertEqual(select_episodes(record, record, ("SIT", "GET_UP")),
                             {"SIT": "0_0", "GET_UP": "0_0"})
            source = root / "episodes.pkl"
            joblib.dump(record, source)
            output = root / "prepared.npz"
            metadata = prepare(source, "0_0", output, root / "assets")
            self.assertEqual(metadata["states_seen"], ["GET_UP", "SIT"])
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data["pos_data"].shape, (2, 24, 3))
                self.assertEqual(data["euler_data"].shape, (2, 24, 3))
                self.assertEqual(data["target_trans"].shape, (2, 3))
            cached = root / "prepared_cached.npz"
            self.assertEqual(prepare(source, "0_0", cached, root / "assets",
                                     record_data=record)["frames"], 2)
            with self.assertRaises(FileExistsError):
                prepare(source, "0_0", output, root / "assets")

    def test_headless_execution_uses_official_state_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "execution"

            class FakeTask:
                def __init__(self):
                    self.state_record = defaultdict(list)
                    self.cur_state = torch.tensor([2, 3])
                    self.is_done = torch.tensor([False, True])
                    self.num_envs = 2
                    self.generator_type = "flow"
                    self.mdm_path = "/real/fm.pt"
                    self.mdm_cfg_param = 7.5
                    self.ode_solver = "rk4"
                    self.ode_steps = 3
                    self.writes = 0

                def _record_states(self):
                    self.state_record["progress"].append(torch.tensor([1, 2]))

                def _write_states_to_file(self, path):
                    self.writes += 1
                    Path(path).write_bytes(b"official writer fixture")

            task = FakeTask()
            recorder = QualityExecutionRecorder(output, steps=2)
            self.assertFalse(recorder.record(task))
            with self.assertRaises(SystemExit) as stopped:
                recorder.record(task)
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(task.writes, 1)
            self.assertEqual(len(task.state_record["task_state"]), 2)
            self.assertTrue((output / "episodes.pkl").is_file())
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["ode_solver"], "rk4")
            with self.assertRaises(FileExistsError):
                QualityExecutionRecorder(output)

    def test_one_case_workflow_writes_curves_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, case_path, output = (root / "manifest.json", root / "case.npz",
                                           root / "smoke")
            manifest.write_text("{}")
            case_path.write_bytes(b"fixture case")
            meta = {"case_id": "REACH_00_normal", "task": "REACH",
                    "difficulty": "normal", "target_joint_names": ["traj"],
                    "heading_target_rad": None, "text": "walk to the target", "seed": 4}
            arrays = {"target_cond": np.zeros((8, 3), dtype=np.float32),
                      "original_target_local_hml_xyz": np.array([1., 0., 0.], dtype=np.float32)}
            records = {label: {"checkpoint": "fixture_" + label,
                               "args_sha256": label} for label in ("dip", "fm")}

            def fake_sample(label, backend, case_meta, case_arrays, device):
                xyz = np.zeros((60, 22, 3), dtype=np.float32)
                stat, curve, conditioned_curve = metrics(
                    xyz, case_arrays["target_cond"], "REACH", ["traj"],
                    actual_target_local=case_arrays["original_target_local_hml_xyz"])
                stat.update({"model": label.upper(), "case_id": case_meta["case_id"],
                             "task": "REACH", "difficulty": "normal", "seed": 4,
                             "cfg": 7.5, "nfe": 10 if label == "dip" else 12})
                return np.zeros((263, 1, 60), dtype=np.float32), xyz, stat, curve, conditioned_curve

            def fake_render(path, *args, **kwargs):
                Path(path).write_bytes(b"fixture media")
                if kwargs.get("gif_path"):
                    Path(kwargs["gif_path"]).write_bytes(b"fixture gif")

            with patch.object(quality_run, "validate_models", return_value=records), \
                    patch.object(quality_run, "load_cases", return_value=[(meta, arrays, case_path)]), \
                    patch.object(quality_run, "load_backend", return_value=None), \
                    patch.object(quality_run, "sample_one", side_effect=fake_sample), \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("quality_render.render_video", side_effect=fake_render), \
                    patch("quality_render.render_distance_curve", side_effect=fake_render):
                self.assertEqual(quality_run.main(["--cases", str(manifest),
                                                   "--output", str(output)]), 0)
            self.assertTrue((output / "summary.md").is_file())
            self.assertTrue((output / "results.csv").is_file())
            with np.load(output / meta["case_id"] / "dip_trajectory.npz") as result:
                self.assertEqual(result["target_distance_curve"].shape, (40, 2))
                self.assertEqual(result["conditioned_target_distance_curve"].shape, (40, 2))

    def test_sample_returns_both_target_distance_curves(self):
        class Raw(torch.nn.Module):
            def forward(self, value):
                return value

        class Model:
            def encode_text(self, text):
                return torch.zeros((len(text), 1))

        class Sampler:
            def p_sample_loop(self, model, shape, **kwargs):
                for _ in range(20):
                    raw(torch.zeros(1))
                return torch.cat((kwargs["model_kwargs"]["y"]["prefix"],
                                  torch.zeros(shape)), dim=-1)

        raw = Raw()
        arrays = {"prefix": np.zeros((263, 1, 20), dtype=np.float32),
                  "mask": np.ones((1, 1, 40), dtype=bool),
                  "target_cond": np.zeros((8, 3), dtype=np.float32),
                  "original_target_local_hml_xyz": np.array([1., 0., 0.], dtype=np.float32),
                  "initial_noise": np.zeros((263, 1, 40), dtype=np.float32),
                  "mean": np.zeros(263, dtype=np.float32),
                  "std": np.ones(263, dtype=np.float32)}
        meta = {"seed": 4, "target_joint_names": ["traj"], "is_heading": False,
                "text": "walk to the target", "task": "REACH",
                "case_id": "REACH_00_normal", "difficulty": "normal"}
        backend = Model(), raw, Sampler(), arrays["mean"], arrays["std"]
        with patch("torch.cuda.synchronize"):
            normalized, xyz, stat, curve, conditioned_curve = sample_one(
                "dip", backend, meta, arrays, torch.device("cpu"))
        self.assertEqual(normalized.shape, (263, 1, 60))
        self.assertEqual(xyz.shape, (60, 22, 3))
        self.assertEqual(stat["mdm_forwards"], 20)
        self.assertEqual(curve.shape, (40, 2))
        self.assertEqual(conditioned_curve.shape, (40, 2))
        self.assertGreater(curve[-1, 1], conditioned_curve[-1, 1])

    def test_capture_can_finish_with_fewer_than_cap_for_one_static_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = QualitySnapshotCapture(Path(temporary) / "conditions",
                                             per_task=16, selected=4)
            counts = dict.fromkeys(TASKS, 16)
            counts["STRIKE_KICK"] = 14
            capture.calls = 9
            capture.last_candidate_call = 5
            self.assertFalse(capture.ready_to_finish(counts))
            capture.calls = 10
            self.assertTrue(capture.ready_to_finish(counts))
            counts["STRIKE_KICK"] = 3
            self.assertFalse(capture.ready_to_finish(counts))

    def test_checkpoint_args_are_separate_and_expected(self):
        models = validate_models()
        self.assertEqual(models["dip"]["args"]["diffusion_steps"], 10)
        self.assertEqual(models["fm"]["args"]["diffusion_steps"], 1000)
        self.assertNotEqual(models["dip"]["args_path"], models["fm"]["args_path"])
        self.assertTrue(models["fm"]["args"]["use_ema"])

    def test_real_capture_serialization_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            n = 20
            rng = np.random.default_rng(123)
            prefixes = torch.tensor(rng.normal(0, 0.01, (n, 263, 1, 20)), dtype=torch.float32)
            goals = torch.zeros(n, 8, 3)
            names = []
            states = []
            for task_name in TASKS:
                joint = {"REACH": "traj", "STRIKE_KICK": "left_foot",
                         "STRIKE_PUNCH": "right_wrist", "SIT": "pelvis",
                         "GET_UP": "pelvis"}[task_name]
                for index in range(4):
                    env = len(states)
                    states.append(int(STATES[task_name]))
                    names.append([joint])
                    goals[env, ("pelvis", "left_foot", "right_foot", "left_wrist",
                                "right_wrist", "head", "traj", "heading").index(joint)] = torch.tensor(
                                    [0.5 + index * 0.3, 0.8, 0.2])
            task = SimpleNamespace(
                num_envs=n, cur_state=torch.tensor(states), progress_buf=torch.arange(n) + 60,
                context_switch_prob=0., planning_horizon_multiplyer=1,
                planning_horizon_30fps=60,
                mean=torch.zeros(263), std=torch.ones(263),
                debug_ends=torch.ones(n, 2, 3), _tar_pos=torch.ones(n, 3),
                pose_buffer=torch.ones(n, 30, 24, 3), generator_type="diffusion",
                _quality_original_target_local=torch.ones(n, 3),
                mdm_path="/real/dip/model000300000.pt")
            y = {"prefix": prefixes, "target_cond": goals,
                 "target_joint_names": names,
                 "is_heading": torch.ones(n, dtype=torch.bool),
                 "text": ["real prompt"] * n,
                 "mask": torch.ones(n, 1, 1, 40, dtype=torch.bool)}
            recon = {"r_rot": torch.tensor([[1., 0, 0, 0]] * n),
                     "r_pos": torch.zeros(n, 3)}
            capture = QualitySnapshotCapture(Path(temporary) / "conditions", per_task=4,
                                             selected=4)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(capture.record(task, {"y": y}, recon))
            manifest_path = Path(temporary) / "conditions/manifest.json"
            cases = load_cases(manifest_path)
            self.assertEqual(len(cases), 20)
            self.assertEqual(set(item[0]["task"] for item in cases), set(TASKS))
            self.assertTrue(all(np.any(item[1]["prefix"]) for item in cases))
            self.assertTrue(all(item[1]["initial_noise"].shape == (263, 1, 40)
                                for item in cases))

    def test_metric_spaces_reach_horizontal_getup_3d_and_no_gt(self):
        xyz = np.zeros((23, 22, 3), dtype=np.float32)
        xyz[20:, 0, 0] = [0, 0.5, 1]
        # Non-degenerate shoulders and hips for heading.
        xyz[:, 2, 0], xyz[:, 1, 0] = 0.3, -0.3
        xyz[:, 17, 0], xyz[:, 16, 0] = 0.4, -0.4
        target = np.zeros((8, 3), dtype=np.float32)
        target[6] = [1, 5, 0]
        target[0] = [1, 5, 0]
        reach, curve, _ = metrics(xyz, target, "REACH", ["traj"])
        getup, _, _ = metrics(xyz, target, "GET_UP", ["pelvis"])
        self.assertAlmostEqual(reach["final_target_error_m"], 0)
        self.assertAlmostEqual(getup["final_target_error_m"], 5)
        original, _, _ = metrics(xyz, target, "REACH", ["traj"],
                                 actual_target_local=np.array([2., 8., 0.]))
        self.assertAlmostEqual(original["final_target_error_m"], 1)
        self.assertAlmostEqual(original["final_conditioned_target_error_m"], 0)
        self.assertEqual(len(curve), 3)
        self.assertIsNone(reach["matched_gt_error_m"])
        self.assertTrue(np.isfinite(list(reach[key] for key in (
            "root_speed_mean_mps", "root_accel_mean_mps2", "seam_root_jump_m"))).all())

    def test_recovery_shape_and_nonfinite_rejection(self):
        motion = np.zeros((263, 1, 60), dtype=np.float32)
        xyz = recover_xyz(motion, np.zeros(263), np.ones(263), prefix_frames=20)
        self.assertEqual(xyz.shape, (60, 22, 3))
        # The model target frame is anchored at the final prefix pose.
        motion[1, 0, :20] = 0.05
        aligned = recover_xyz(motion, np.zeros(263), np.ones(263), prefix_frames=20)
        self.assertAlmostEqual(float(aligned[19, 0, 0]), 0, places=5)
        self.assertGreater(abs(float(aligned[0, 0, 0])), 0.1)
        motion[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            recover_xyz(motion, np.zeros(263), np.ones(263))

    def test_minimal_headless_mp4_and_gif_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = np.zeros((3, 22, 3), dtype=np.float32)
            base[:, :, 1] = np.linspace(0.0, 1.5, 22)
            fm = base.copy()
            fm[:, :, 0] += 0.1
            mp4 = Path(temporary) / "side.mp4"
            gif = Path(temporary) / "side.gif"
            render_video(mp4, base, fm, np.array([1., 0., 0.]),
                         "CPU renderer fixture", prefix_frames=1, fps=20,
                         side_by_side=True, gif_path=gif)
            verify_media(mp4)
            verify_media(gif)
            self.assertGreater(mp4.stat().st_size, 1024)
            self.assertGreater(gif.stat().st_size, 1024)

    def test_summary_groups_all_metrics_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "provenance.json").write_text(json.dumps({"cases_requested": 1}))
            case = run / "REACH_00_normal"
            case.mkdir()
            for name in ("dip_trajectory.npz", "fm_trajectory.npz",
                         "dip_target_distance.csv", "fm_target_distance.csv",
                         "target_distance.png", "dip.mp4", "fm.mp4",
                         "side_by_side.mp4"):
                (case / name).write_bytes(b"fixture")
            for model in ("dip", "fm"):
                data = {key: 1.0 if model == "dip" else 2.0 for key in summary.NUMERIC}
                data.update({"case_id": case.name, "task": "REACH", "difficulty": "normal",
                             "model": model.upper(), "seed": 10000, "cfg": 7.5,
                             "case_input_sha256": "same", "target_metric": "horizontal_pelvis",
                             "target_joint": "traj", "target_metric_note": "fixture",
                             "matched_gt_error_m": None,
                             "trajectory_npz": str(Path(case.name) / (model + "_trajectory.npz")),
                             "distance_curve_csv": str(Path(case.name) / (model + "_target_distance.csv")),
                             "distance_curve_png": str(Path(case.name) / "target_distance.png"),
                             "model_mp4": str(Path(case.name) / (model + ".mp4")),
                             "comparison_mp4": str(Path(case.name) / "side_by_side.mp4"),
                             "comparison_gif": None})
                (case / (model + "_metrics.json")).write_text(json.dumps(data))
            summary.summarize(run)
            self.assertIn("SMOKE COMPLETE", (run / "summary.md").read_text())
            self.assertIn("root_accel_mean_mps2_mean", (run / "group_summary.csv").read_text())
            with self.assertRaises(FileExistsError):
                summary.summarize(run)


if __name__ == "__main__":
    unittest.main()
