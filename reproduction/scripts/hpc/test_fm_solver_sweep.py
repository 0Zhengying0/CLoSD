"""CPU-only numerical and conditioning checks for the FM solver sweep."""

import math
import contextlib
import io
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from closd.diffusion_planner.diffusion.ode_sampler import VelocityODESampler
from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
import summarize_fm_solver_sweep as summary


class RecordingField(torch.nn.Module):
    def __init__(self, field):
        super().__init__()
        self.field = field
        self.times = []

    def forward(self, x, timesteps, y=None):
        self.times.append(int(timesteps[0]))
        assert timesteps.dtype == torch.long
        assert torch.all(timesteps == timesteps[0])
        return self.field(x, timesteps)


class MutatingMDM(torch.nn.Module):
    """Reproduce MDM's y['mask'] rebinding and the two CFG branches."""

    cond_mask_prob = 0.1
    cond_mode = "text"

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, timesteps, y=None):
        self.seen.append((
            int(timesteps[0]),
            y["mask"].shape[-1],
            y["prefix"].clone(),
            y["target_cond"].clone(),
            bool(y.get("text_uncond", False)),
        ))
        y["mask"] = torch.cat((torch.ones_like(y["mask"]), y["mask"]), dim=-1)
        y["transient"] = True
        return torch.full_like(x, 0.0 if y.get("text_uncond", False) else 2.0)


class SolverSweepTests(unittest.TestCase):
    def sample(self, solver, steps, model, x0, model_kwargs=None, timesteps=1001):
        sampler = VelocityODESampler(
            num_timesteps=timesteps, num_steps=steps, solver=solver,
        )
        result = sampler.p_sample_loop(
            model, tuple(x0.shape), noise=x0.clone(), model_kwargs=model_kwargs,
            device=x0.device,
        )
        return sampler, result

    def test_analytic_exponential_and_nfe(self):
        x0 = torch.ones((2, 3, 1, 4), dtype=torch.float64)
        errors = {}
        for solver, multiplier in (("euler", 1), ("heun", 2), ("rk4", 4)):
            field = RecordingField(lambda x, _t: x)
            sampler, result = self.sample(solver, 6, field, x0)
            self.assertEqual(sampler.nfe, 6 * multiplier)
            self.assertEqual(len(field.times), sampler.nfe)
            self.assertEqual(result.shape, x0.shape)
            errors[solver] = abs(result[0, 0, 0, 0].item() - math.e)
        self.assertLess(errors["rk4"], errors["heun"])
        self.assertLess(errors["heun"], errors["euler"])
        self.assertLess(errors["rk4"], 3e-5)

    def test_time_grid_endpoints_midpoints_and_shapes(self):
        x0 = torch.zeros((2, 3, 1, 4), dtype=torch.float64)
        for solver, count in (("euler", 4), ("heun", 8), ("rk4", 16)):
            field = RecordingField(lambda x, _t: torch.ones_like(x))
            sampler, result = self.sample(solver, 4, field, x0)
            self.assertEqual(sampler.nfe, count)
            self.assertEqual(result.shape, x0.shape)
            self.assertTrue(torch.allclose(result, torch.ones_like(result)))
            self.assertEqual(field.times[0], 0)
            if solver != "euler":
                self.assertEqual(field.times[-1], 1000)
            else:
                self.assertEqual(field.times[-1], 750)
        field = RecordingField(lambda x, _t: torch.ones_like(x))
        self.sample("rk4", 4, field, x0)
        self.assertEqual(field.times, [
            0, 125, 125, 250,
            250, 375, 375, 500,
            500, 625, 625, 750,
            750, 875, 875, 1000,
        ])

    def test_discrete_time_conversion_for_nonexact_midpoints(self):
        x0 = torch.zeros((1, 1, 1, 1))
        field = RecordingField(lambda x, _t: torch.ones_like(x))
        sampler, _ = self.sample("rk4", 3, field, x0, timesteps=1000)
        grid = torch.linspace(0, 1, 4)
        expected = []
        for a, b in zip(grid[:-1], grid[1:]):
            t0, t1 = a.item(), b.item()
            expected.extend([
                sampler._time_to_index(t0, 1, x0.device).item(),
                sampler._time_to_index(t0 + (t1 - t0) / 2, 1, x0.device).item(),
                sampler._time_to_index(t0 + (t1 - t0) / 2, 1, x0.device).item(),
                sampler._time_to_index(t1, 1, x0.device).item(),
            ])
        self.assertEqual(field.times, expected)
        self.assertEqual(field.times[0], 0)
        self.assertEqual(field.times[-1], 999)
        self.assertTrue(all(a <= b for a, b in zip(field.times, field.times[1:])))

    def test_euler_and_heun_reference_formulas(self):
        x0 = torch.full((1, 2, 1, 3), 0.7, dtype=torch.float64)
        for solver in ("euler", "heun"):
            field = RecordingField(
                lambda x, t: 0.2 * x + t.to(x.dtype).view(-1, 1, 1, 1) / 1000,
            )
            _, actual = self.sample(solver, 5, field, x0)
            expected = x0.clone()
            grid = torch.linspace(0, 1, 6)
            for a, b in zip(grid[:-1], grid[1:]):
                dt = b.item() - a.item()
                k0 = 0.2 * expected + round(a.item() * 1000) / 1000
                if solver == "euler":
                    expected = expected + dt * k0
                else:
                    pred = expected + dt * k0
                    k1 = 0.2 * pred + round(b.item() * 1000) / 1000
                    expected = expected + 0.5 * dt * (k0 + k1)
            self.assertTrue(torch.equal(actual, expected), solver)

    def test_cfg_conditions_prefix_and_actual_raw_forwards(self):
        x0 = torch.zeros((2, 3, 1, 4))
        prefix = torch.arange(12, dtype=x0.dtype).reshape(2, 3, 1, 2)
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.zeros((2, 1, 1, 4), dtype=torch.bool)
        kwargs = {"y": {
            "prefix": prefix.clone(), "mask": mask.clone(),
            "target_cond": target.clone(), "text": ["one", "two"],
            "scale": torch.full((2,), 7.5),
        }}
        for solver, multiplier in (("euler", 1), ("heun", 2), ("rk4", 4)):
            raw = MutatingMDM()
            cfg_model = ClassifierFreeSampleModel(raw)
            self.assertIs(cfg_model.model, raw)
            forwards = [0]
            hook = raw.register_forward_hook(
                lambda _module, _inputs, _output: forwards.__setitem__(0, forwards[0] + 1),
            )
            try:
                sampler, result = self.sample(solver, 2, cfg_model, x0, kwargs)
            finally:
                hook.remove()
            self.assertEqual(forwards[0], 2 * sampler.nfe)
            self.assertEqual(len(raw.seen), forwards[0])
            self.assertEqual(result.shape, (2, 3, 1, 6))
            self.assertTrue(torch.equal(result[..., :2], prefix))
            self.assertTrue(torch.allclose(result[..., 2:], torch.full_like(x0, 15.0)))
            self.assertEqual(kwargs["y"]["mask"].shape[-1], 4)
            self.assertNotIn("transient", kwargs["y"])
            self.assertTrue(torch.equal(kwargs["y"]["mask"], mask))
            self.assertTrue(torch.equal(kwargs["y"]["prefix"], prefix))
            self.assertTrue(torch.equal(kwargs["y"]["target_cond"], target))
            for _, mask_length, seen_prefix, seen_target, _ in raw.seen:
                self.assertEqual(mask_length, 4)
                self.assertTrue(torch.equal(seen_prefix, prefix))
                self.assertTrue(torch.equal(seen_target, target))
            self.assertEqual(sum(seen[-1] for seen in raw.seen), sampler.nfe)

    def test_summary_uses_exact_counts_and_marks_missing_or_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = "/checkpoints/model000300000.pt"
            rows = []
            for experiment_id, solver, steps, nfe in (
                ("FM-E12", "euler", 12, 12),
                ("FM-H12", "heun", 6, 12),
                ("FM-R12", "rk4", 3, 12),
            ):
                rows.append({
                    "experiment_id": experiment_id,
                    "solver": solver, "steps": str(steps), "nfe": str(nfe),
                    "job_id": "123" if experiment_id == "FM-E12" else "124",
                    "submit_status": "submitted",
                    "stdout": str(directory / (experiment_id + "_%j.out")),
                    "stderr": str(directory / (experiment_id + "_%j.err")),
                    "checkpoint": checkpoint,
                })
            successes = (731, 851, 851, 941, 691)
            resets = (1001, 1000, 1000, 1000, 1000)
            lines = [
                "SWEEP_EXPERIMENT_ID=FM-E12", "SWEEP_JOB_ID=123",
                "SWEEP_CHECKPOINT=" + checkpoint,
                "SWEEP_GIT_COMMIT=abc123", "SWEEP_GIT_DIRTY=1",
                "SWEEP_SOURCE_SHA256=abcdef", "SWEEP_SEED=0",
                "SWEEP_CONFIG=cfg:7.5",
                "PLANNER_LATENCY_SEC=12.000000 GENERATOR=flow ODE_STEPS=12 SOLVER=euler NFE=12 MODEL_FORWARD_CALLS=24 BATCH=4096",
                "PLANNER_LATENCY_SEC=10.000000 GENERATOR=flow ODE_STEPS=12 SOLVER=euler NFE=12 MODEL_FORWARD_CALLS=24 BATCH=4096",
            ]
            lines.extend(
                "EVAL_RESULT STATE={} SUCCESS={} RESETS={} RATE={:.8f}".format(
                    state, success, reset, success / reset,
                )
                for state, success, reset in zip(summary.ORDER, successes, resets)
            )
            lines.extend((
                "EVAL_MACRO_SUCCESS={:.8f}".format(
                    statistics.mean(s / n for s, n in zip(successes, resets)),
                ),
                "EVAL_PROFILE CALLS=2 MODEL_FORWARDS_TOTAL=48 "
                "PLANNER_LATENCY_TOTAL_SEC=22.000000 PLANNER_LATENCY_MEAN_SEC=11.000000",
                "Eval plot ended", "SWEEP_PROCESS_EXIT=0", "SWEEP_JOB_EXIT=0",
            ))
            Path(rows[0]["stdout"].replace("%j", "123")).write_text("\n".join(lines))
            Path(rows[2]["stdout"].replace("%j", "124")).write_text(
                "Traceback (most recent call last):\nSWEEP_PROCESS_EXIT=1\n",
            )
            self.assertEqual(summary.inspect_job(rows[0])["status"], "COMPLETE")
            self.assertEqual(summary.inspect_job(rows[1])["status"], "MISSING_LOG")
            self.assertEqual(summary.inspect_job(rows[2])["status"], "FAILED")
            markdown = summary.render(rows, directory)
            exact_macro = statistics.mean(s / n for s, n in zip(successes, resets))
            self.assertIn("{:.2f}%".format(100 * exact_macro), markdown)
            self.assertIn("11.000 / 11.000", markdown)
            self.assertIn("MISSING_LOG", markdown)
            self.assertIn("FAILED", markdown)
            self.assertIn("| FM-H12 | MISSING_LOG | heun", markdown)

    def test_summary_cli_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = directory / "submissions.tsv"
            ids = ("FM-E12", "FM-H12", "FM-R12", "FM-E20", "FM-H20",
                   "FM-R20", "FM-E40", "FM-H40", "FM-R40")
            manifest.write_text(
                "experiment_id\tsolver\tsteps\tnfe\tjob_id\tstdout\tstderr\t"
                "submit_status\tcheckpoint\n" +
                "".join("{}\teuler\t12\t12\t\tunused\tunused\tsubmit_failed\tckpt\n".format(item)
                        for item in ids)
            )
            output = directory / "summary.md"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(summary.main([
                    "--run-dir", str(directory), "--output", str(output),
                ]), 0)
            self.assertIn("SUBMIT_FAILED", output.read_text())
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    summary.main([
                        "--run-dir", str(directory), "--output", str(output),
                    ])

    def test_slurm_rk4_preflight_accepts_only_the_declared_nfe(self):
        script = ROOT / "reproduction/scripts/hpc/closd_velocity_eval.slurm"
        checkpoint = ROOT / (
            "reproduction/runs/FlowMatching_DIMP_FINAL_seed10_300k/"
            "model000300000.pt"
        )
        environment = dict(os.environ, GENERATOR="flow", MODEL=str(checkpoint),
                           ODE_SOLVER="rk4", ODE_STEPS="3",
                           EXPERIMENT_ID="FM-R12",
                           SWEEP_RUN_ROOT=str(ROOT / "reproduction/experiments/fm_solver_sweep/runs/preflight_test"))
        good = subprocess.run(
            ["bash", str(script), "--check"], cwd=str(ROOT), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        )
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertIn("NFE:       12", good.stdout)
        environment["ODE_STEPS"] = "4"
        bad = subprocess.run(
            ["bash", str(script), "--check"], cwd=str(ROOT), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        )
        self.assertEqual(bad.returncode, 2)
        self.assertIn("requires rk4 and NFE=12", bad.stderr)

    def test_multiprocessing_socket_path_budget(self):
        """A future TMPDIR change must not reproduce the nine-job startup failure."""
        script = (ROOT / "reproduction/scripts/hpc/closd_velocity_eval.slurm").read_text()
        self.assertIn("export TMPDIR=$(mktemp -d /tmp/cfm.XXXXXXXX)", script)
        self.assertIn("s.bind(p)", script)
        # Python multiprocessing adds its own pymp directory and listener name.
        longest_expected = "/tmp/cfm." + "x" * 8 + "/pymp-" + "x" * 8 + "/listener-" + "x" * 8
        self.assertLess(len(os.fsencode(longest_expected)), 107)


if __name__ == "__main__":
    unittest.main()
