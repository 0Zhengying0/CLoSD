"""CPU-only checks for the optional closed-loop episode diagnostic path."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from closd.learning.episode_diagnostics import EpisodeDiagnosticRecorder, snapshot_from_task
from closd.utils.closd_util import STATES
import analyze_episode_diagnostics as analysis


METRICS = ("REACH", "STRIKE_KICK", "STRIKE_PUNCH", "SIT", "GET_UP")


def official_masks(done, terminated, task_done, state):
    successful = done & ~terminated & task_done
    strike = np.isin(state, (STATES.STRIKE_KICK, STATES.STRIKE_PUNCH))
    strike_mask = {
        "counted": done & (strike | (state == STATES.HALT)),
        "success": (successful & strike) | (done & (state == STATES.HALT)),
    }
    return {
        "REACH": {"counted": done & (state == STATES.REACH),
                  "success": successful & (state == STATES.REACH)},
        "STRIKE_KICK": strike_mask,
        "STRIKE_PUNCH": strike_mask,
        "SIT": {"counted": done & np.isin(state, (STATES.SIT, STATES.GET_UP)),
                "success": (successful & (state == STATES.SIT)) | (done & (state == STATES.GET_UP))},
        "GET_UP": {"counted": done & (state == STATES.GET_UP),
                   "success": successful & (state == STATES.GET_UP)},
    }


def make_snapshot(step, states, initials, hits, latched, terminated, done):
    states = np.asarray(states, dtype=np.int16)
    initials = np.asarray(initials, dtype=np.int16)
    hits = np.asarray(hits, dtype=np.bool_)
    latched = np.asarray(latched, dtype=np.bool_)
    terminated = np.asarray(terminated, dtype=np.bool_)
    done = np.asarray(done, dtype=np.bool_)
    n = len(states)
    body = np.zeros((n, 2, 3), dtype=np.float32)
    body[:, 0, 2] = 0.93 if step == 4 else 0.4
    return {
        "progress_step": np.full(n, step, dtype=np.int32),
        "state": states, "initial_state": initials,
        "body_pos_xyz": body,
        "controller_action": np.full((n, 3), step / 10, dtype=np.float32),
        "pd_target": np.full((n, 3), step / 5, dtype=np.float32),
        "dof_pos": np.full((n, 3), step / 6, dtype=np.float32),
        "target_xyz": np.ones((n, 3), dtype=np.float32),
        "target_actor_pose": np.tile(np.array([1, 2, 3, 0, 0, 0, 1], dtype=np.float32), (n, 1)),
        "goal_distance_m": np.where(hits, 0.0, 0.5).astype(np.float32),
        "box_up_dot": np.full(n, np.nan, dtype=np.float32),
        "physical_goal_hit": hits,
        "official_task_done": latched,
        "terminated": terminated,
        "done": done,
    }


class DiagnosticTests(unittest.TestCase):
    def build_finished_recorder(self, directory):
        recorder = EpisodeDiagnosticRecorder(
            directory, num_envs=2, max_steps=5, dt=1 / 30,
            checkpoint="/models/checkpoint.pt", generator="flow",
            metric_names=METRICS, shard_size=1,
        )
        initials = [STATES.SIT, STATES.REACH]
        records = [
            make_snapshot(1, [STATES.SIT, STATES.REACH], initials,
                          [True, False], [False, False], [False, False], [False, False]),
            make_snapshot(2, [STATES.SIT, STATES.REACH], initials,
                          [False, True], [True, False], [False, False], [False, False]),
            make_snapshot(3, [STATES.GET_UP, STATES.REACH], initials,
                          [False, False], [False, True], [False, False], [False, False]),
            make_snapshot(4, [STATES.GET_UP, STATES.REACH], initials,
                          [True, False], [False, True], [True, False], [True, True]),
        ]
        counters = {STATES[name]: {"resets": 0, "success": 0} for name in METRICS}
        for snapshot in records:
            masks = official_masks(
                snapshot["done"], snapshot["terminated"],
                snapshot["official_task_done"], snapshot["state"],
            )
            recorder.record_step(snapshot, masks)
            for name in METRICS:
                counters[STATES[name]]["resets"] += int(masks[name]["counted"].sum())
                counters[STATES[name]]["success"] += int(masks[name]["success"].sum())
        recorder.finish(records[-1], counters)
        return recorder

    def test_episode_events_and_original_official_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "diagnostics"
            recorder = self.build_finished_recorder(directory)
            manifest, rows = analysis.read_diagnostics(directory)
            self.assertEqual(manifest["completed_episodes"], 2)
            self.assertEqual(manifest["incomplete_episodes"], 0)
            self.assertEqual(recorder.metric_counts["SIT"], {"resets": 1, "success": 1})
            sit = next(row for row in rows if row["initial_task"] == "SIT")
            self.assertEqual(sit["first_initial_goal_step"], 1)
            self.assertEqual(sit["first_official_task_done_step"], 2)
            self.assertEqual(sit["sit_to_get_up_step"], 2)
            self.assertEqual(sit["get_up_first_goal_step"], 4)
            self.assertEqual(sit["fall_step"], 4)
            self.assertEqual(sit["terminal_reason"], "fall")
            self.assertTrue(sit["official_success_for_initial_task"])
            self.assertFalse(sit["official_get_up_success"])
            self.assertEqual(sit["post_get_up_final_state"], "GET_UP")
            self.assertTrue(sit["post_get_up_fell"])
            self.assertEqual(sit["state_transitions"][0]["to"], "GET_UP")
            self.assertEqual(sit["trajectory_steps"], 4)
            reach = next(row for row in rows if row["initial_task"] == "REACH")
            self.assertEqual(reach["terminal_reason"], "time_limit")
            self.assertTrue(reach["official_success_for_initial_task"])

    def test_full_control_trajectory_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "diagnostics"
            self.build_finished_recorder(directory)
            csv_path = Path(temporary) / "trace.csv"
            self.assertEqual(analysis.export_trajectory(directory, "env0000-ep0000", csv_path), 4)
            header = csv_path.read_text().splitlines()[0]
            self.assertIn("body_1_z", header)
            self.assertIn("action_2", header)
            self.assertIn("pd_target_2", header)
            self.assertIn("dof_pos_2", header)
            with self.assertRaises(FileExistsError):
                analysis.export_trajectory(directory, "env0000-ep0000", csv_path)

    def test_comparison_and_episode_csv_mark_missing_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.build_finished_recorder(run_dir / "fm_rk4_3" / "diagnostics")
            markdown = analysis.render_summary(run_dir)
            self.assertIn("FM-RK4-3 | COMPLETE", markdown)
            self.assertIn("DiP-original | MISSING/FAILED", markdown)
            self.assertIn("SIT → GET_UP", markdown)
            csv_path = run_dir / "episodes.csv"
            self.assertEqual(analysis.export_episodes_csv(run_dir, csv_path), 2)
            self.assertIn("official_success_for_initial_task", csv_path.read_text())

    def test_report_does_not_call_a_failed_slurm_job_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            self.build_finished_recorder(run_dir / "fm_rk4_3" / "diagnostics")
            log = run_dir / "fm.out"
            log.write_text("DIAG_PROCESS_EXIT=1\nDIAG_JOB_EXIT=1\n")
            (run_dir / "submissions.tsv").write_text(
                "label\tgenerator\tsteps\tmodel\tjob_id\tsubmit_status\tstdout\tstderr\n"
                "FM-RK4-3\tflow\t3\tmodel\t123\tsubmitted\t{}\tunused\n"
                "DIP-ORIGINAL\tdiffusion\t0\tmodel\t\tsubmit_failed\tunused\tunused\n".format(log)
            )
            report = analysis.render_summary(run_dir)
            self.assertIn("FM-RK4-3 | FAILED", report)
            self.assertIn("DiP-original | SUBMIT_FAILED", report)
            self.assertNotIn("FM-RK4-3 | COMPLETE", report)

    def test_unfinished_episode_has_no_official_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "diagnostics"
            recorder = EpisodeDiagnosticRecorder(
                directory, 2, 5, 1 / 30, "checkpoint", "flow", METRICS,
            )
            snapshot = make_snapshot(
                1, [STATES.REACH, STATES.SIT], [STATES.REACH, STATES.SIT],
                [True, False], [False, False], [False, False], [True, False],
            )
            masks = official_masks(snapshot["done"], snapshot["terminated"],
                                   snapshot["official_task_done"], snapshot["state"])
            recorder.record_step(snapshot, masks)
            counters = {STATES[name]: {"resets": int(masks[name]["counted"].sum()),
                                       "success": int(masks[name]["success"].sum())}
                        for name in METRICS}
            recorder.finish(snapshot, counters)
            manifest, rows = analysis.read_diagnostics(directory)
            self.assertEqual(manifest["completed_episodes"], 1)
            self.assertEqual(manifest["incomplete_episodes"], 1)
            partial = next(row for row in rows if not row["completed"])
            self.assertEqual(partial["terminal_reason"], "evaluation_stopped")
            self.assertIsNone(partial["official_success_for_initial_task"])

    def test_strike_to_halt_then_fall_keeps_the_original_official_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "diagnostics"
            recorder = EpisodeDiagnosticRecorder(
                directory, 1, 5, 1 / 30, "checkpoint", "flow", METRICS,
            )
            initials = [STATES.STRIKE_KICK]
            snapshots = [
                make_snapshot(1, [STATES.STRIKE_KICK], initials,
                              [True], [False], [False], [False]),
                make_snapshot(2, [STATES.HALT], initials,
                              [False], [False], [True], [True]),
            ]
            counters = {STATES[name]: {"resets": 0, "success": 0} for name in METRICS}
            for snapshot in snapshots:
                masks = official_masks(snapshot["done"], snapshot["terminated"],
                                       snapshot["official_task_done"], snapshot["state"])
                recorder.record_step(snapshot, masks)
                for name in METRICS:
                    counters[STATES[name]]["resets"] += int(masks[name]["counted"].sum())
                    counters[STATES[name]]["success"] += int(masks[name]["success"].sum())
            recorder.finish(snapshots[-1], counters)
            _manifest, rows = analysis.read_diagnostics(directory)
            row = rows[0]
            self.assertEqual(row["initial_task"], "STRIKE_KICK")
            self.assertEqual(row["first_initial_goal_step"], 1)
            self.assertEqual(row["fall_step"], 2)
            self.assertTrue(row["official_success_for_initial_task"])
            self.assertTrue(row["official_metric_flags"]["STRIKE_PUNCH"]["success"])
            self.assertIsNone(row["first_official_task_done_step"])

    def test_counter_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = EpisodeDiagnosticRecorder(
                Path(temporary) / "diagnostics", 1, 5, 1 / 30,
                "checkpoint", "flow", METRICS,
            )
            snapshot = make_snapshot(1, [STATES.REACH], [STATES.REACH],
                                     [True], [True], [False], [True])
            masks = official_masks(snapshot["done"], snapshot["terminated"],
                                   snapshot["official_task_done"], snapshot["state"])
            recorder.record_step(snapshot, masks)
            bad = {STATES[name]: {"resets": 0, "success": 0} for name in METRICS}
            with self.assertRaisesRegex(RuntimeError, "counter mismatch"):
                recorder.finish(snapshot, bad)

    def test_snapshot_task_specific_distances_and_predicates(self):
        class FakeTask:
            pass

        task = FakeTask()
        task._rigid_body_pos = torch.tensor([
            [[0.1, 0.0, 0.9], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.4], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.9], [1.0, 0.0, 0.9]],
            [[1.0, 0.0, 0.93], [0.0, 0.0, 0.0]],
        ])
        task._tar_pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.4],
                                      [0.0, 0.0, 0.9], [1.0, 0.0, 0.93]])
        task._target_states = torch.tensor([
            [0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0.9, 0.7071068, 0., 0., 0.7071068],
            [0., 0., 0., 0., 0., 0., 1.],
        ])
        task.mujoco_joint_idx = torch.tensor([[0], [0], [1], [0]])
        task.bench_corners = torch.tensor([[
            [1., 1., .3], [-1., 1., .3], [-1., -1., .3], [1., -1., .3]
        ]] * 4)
        task.sit_dims = np.array([2., 2., .3])
        task.max_gap_from_sit = .2
        task.actions = torch.ones((4, 3))
        task._episode_diag_pd_target = torch.ones((4, 3)) * 2
        task._dof_pos = torch.ones((4, 3)) * 3
        task.progress_buf = torch.ones(4, dtype=torch.int32)
        task.init_state_per_env = torch.tensor([
            STATES.REACH, STATES.SIT, STATES.STRIKE_KICK, STATES.SIT,
        ])
        task.get_cur_done = lambda: torch.tensor([True, True, True, True])
        states = torch.tensor([STATES.REACH, STATES.SIT, STATES.STRIKE_KICK, STATES.GET_UP])
        info = {"task_state": states, "task_done": torch.zeros(4, dtype=torch.bool),
                "terminate": torch.zeros(4, dtype=torch.bool)}
        snap = snapshot_from_task(task, info, torch.zeros(4, dtype=torch.bool))
        self.assertAlmostEqual(float(snap["goal_distance_m"][0]), .1, places=5)
        self.assertAlmostEqual(float(snap["goal_distance_m"][1]), 0, places=5)
        self.assertAlmostEqual(float(snap["goal_distance_m"][2]), 0, places=5)
        self.assertAlmostEqual(float(snap["goal_distance_m"][3]), 0, places=5)
        self.assertAlmostEqual(float(snap["box_up_dot"][2]), 0, places=5)

    def test_slurm_preflight_both_samplers_and_preserves_dip_symlink(self):
        script = ROOT / "reproduction/scripts/hpc/closd_velocity_eval.slurm"
        models = (
            ("FM-RK4-3", "flow", "rk4", "3",
             ROOT / "reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/model000300000.pt"),
            ("DIP-ORIGINAL", "diffusion", "diffusion", "0",
             ROOT / "closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for label, generator, solver, steps, model in models:
                environment = dict(os.environ, GENERATOR=generator, MODEL=str(model),
                                   ODE_SOLVER=solver, ODE_STEPS=steps,
                                   DIAG_LABEL=label, DIAG_RUN_ROOT=str(Path(temporary) / label),
                                   EXPERIMENT_ID="", SWEEP_RUN_ROOT="")
                checked = subprocess.run(
                    ["bash", str(script), "--check"], cwd=str(ROOT), env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                )
                self.assertEqual(checked.returncode, 0, checked.stderr)
                self.assertIn("Model:     " + str(model), checked.stdout)
            environment["DIAG_LABEL"] = "FM-RK4-3"
            checked = subprocess.run(
                ["bash", str(script), "--check"], cwd=str(ROOT), env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
            )
            self.assertEqual(checked.returncode, 2)
            self.assertIn("requires flow, rk4, three steps", checked.stderr)


if __name__ == "__main__":
    unittest.main()
